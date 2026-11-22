import io
import threading
import time

from antigravity_swap.settings import Settings
from antigravity_swap.store import Registry
from antigravity_swap.supervise import LogWatch, resume_args, run_interactive

from test_core import add_accounts


def run(env, args, **kw):
    err = io.StringIO()
    rc = run_interactive(env.manager(), args, err=err, grace=0, **kw)
    return rc, err.getvalue()


def test_retry_lines_do_not_trigger_but_final_error_does(tmp_path):
    w = LogWatch(tmp_path / "x.log")
    w.feed("I1002 run.go:371] Run: attempt 1 failed (RESOURCE_EXHAUSTED (code 429): Individual quota reached. "
           "Resets in 2h44m0s.), retrying in 1s\n")
    assert not w.hit.is_set()
    w.feed("I1002 server.go:1248] Created conversation 7d5d4ba2-f44e-4762-9dfe-33bd01518022\n")
    w.feed("E1002 errorreport.go:223] agent executor error: calling model: RESOURCE_EXHAUSTED (code 429): "
           "Individual quota reached. Resets in 2h43m56s.\n")
    assert w.hit.is_set()
    assert w.conv == "7d5d4ba2-f44e-4762-9dfe-33bd01518022"


def test_resume_args():
    assert resume_args(["--model", "m", "-c"], "abc", "go on") == \
        ["--model", "m", "--conversation", "abc", "-i", "go on"]


def test_session_moves_to_next_account_and_continues(env, capfd):
    env.plan({"a@x.io": "quota", "b@x.io": "ok", "c@x.io": "ok"})
    add_accounts(env, "a@x.io", "b@x.io", "c@x.io")
    start = time.time()
    rc, err = run(env, ["--model", "gemini-3.8-flash-high"], target="1", strategy="rotate")
    out = capfd.readouterr().out
    assert rc == 0, err
    assert time.time() - start < 30  # the hung session on account 1 was stopped, not waited out
    assert "session by b@x.io" in out and "turns=2" in out  # same conversation, one turn from each account
    assert "prompt='Please continue where you left off.'" in out
    a = Registry.load().get(1)
    assert a.is_limited() and abs(a.limited_until - (time.time() + 1800)) < 60
    assert "held out until" in err and "continuing conversation" in err


def test_session_reuses_recovered_accounts_beyond_one_round(env, monkeypatch, capfd):
    from antigravity_swap import supervise

    env.plan({"a@x.io": "quota", "b@x.io": "quota"})
    add_accounts(env, "a@x.io", "b@x.io")
    real_flag, real_next, real_popen = supervise._flag, supervise._next, supervise.subprocess.Popen
    procs, checks = [], []

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        procs.append(proc)
        return proc

    def recovered(slot, limited=None, quarantine=None):
        # Quota windows elapse between hops in a long-running real conversation.
        real_flag(slot, limited=(time.time() - 1, limited[1]))

    def next_account(mgr, strategy, pool, excluded, after):
        checks.append(after)
        assert excluded == {after}  # an earlier account must not be excluded forever
        if len(checks) == 3:
            env.plan({"a@x.io": "quota", "b@x.io": "ok"})
        return real_next(mgr, strategy, pool, excluded, after)

    monkeypatch.setattr(supervise.subprocess, "Popen", popen)
    monkeypatch.setattr(supervise, "_flag", recovered)
    monkeypatch.setattr(supervise, "_next", next_account)
    watchdog = threading.Timer(10, lambda: [p.kill() for p in procs if p.poll() is None])
    watchdog.start()
    try:
        rc, err = run(env, [], target="1", strategy="rotate")
        assert rc == 0, err
        assert checks == [1, 2, 1]  # 1 -> 2 -> recovered 1 -> recovered 2
        assert "turns=4" in capfd.readouterr().out
        assert err.count("continuing conversation") == 3
    finally:
        watchdog.cancel()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def test_waits_and_retries_when_another_account_recovers(env, monkeypatch, capfd):
    from antigravity_swap import supervise

    env.plan({"a@x.io": "quota", "b@x.io": "ok"})
    add_accounts(env, "a@x.io", "b@x.io")
    real_next = supervise._next
    checks = []

    def next_account(*args):
        checks.append(time.monotonic())
        return None if len(checks) == 1 else real_next(*args)

    monkeypatch.setattr(supervise, "_next", next_account)
    rc, err = run(env, [], target="1", strategy="rotate", retry_interval=0.05)
    assert rc == 0, err
    assert len(checks) == 2  # only one quota error was logged by the first process
    assert "checking again every" in err
    assert "turns=2" in capfd.readouterr().out


def test_explicit_hop_limit_is_still_enforced(env, monkeypatch, capfd):
    from antigravity_swap import supervise

    env.plan({"a@x.io": "quota", "b@x.io": "quota", "c@x.io": "ok"})
    add_accounts(env, "a@x.io", "b@x.io", "c@x.io")
    mgr = env.manager()
    mgr.settings = Settings({"failover.max_hops": 1})
    real_popen = supervise.subprocess.Popen
    procs = []

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        procs.append(proc)
        return proc

    monkeypatch.setattr(supervise.subprocess, "Popen", popen)
    err = io.StringIO()
    watchdog = threading.Timer(4, lambda: [p.kill() for p in procs if p.poll() is None])
    watchdog.start()
    try:
        run_interactive(mgr, [], target="1", strategy="rotate", err=err, grace=0)
        assert len(procs) == 2  # account 3 was available but the explicit cap stopped hopping
        assert "failover.max_hops (1) reached" in err.getvalue()
    finally:
        watchdog.cancel()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def test_log_watch_distinguishes_a_new_turn_from_a_failed_turn(tmp_path):
    watch = LogWatch(tmp_path / "x.log")
    watch.feed("agent executor error: Individual quota reached. Resets in 1h\n")
    assert watch.quota_turn == watch.turn
    watch.feed("Sending user message to conversation 7d5d4ba2-f44e-4762-9dfe-33bd01518022\n")
    assert watch.quota_turn != watch.turn


def test_stays_put_when_no_account_is_free(env, monkeypatch, capfd):
    env.plan({"a@x.io": "quota"})
    add_accounts(env, "a@x.io")
    # one account only: aswap must leave the session alone (the fake exits by itself after 60s,
    # so shorten that by letting the test kill it via max wait)
    import threading
    from antigravity_swap import supervise

    procs = []
    real_popen = supervise.subprocess.Popen

    def popen(*a, **k):
        p = real_popen(*a, **k)
        procs.append(p)
        threading.Timer(4, p.kill).start()
        return p

    monkeypatch.setattr(supervise.subprocess, "Popen", popen)
    rc, err = run(env, [])
    assert "no other account is free; staying on it" in err
    assert len(procs) == 1


def test_strategy_aliases():
    s = Settings()
    assert s.set("failover.strategy", "round-robin") == "rotate"
    assert s.set("failover.strategy", "most-quota") == "best"
    assert s.set("autoswitch.strategy", "soonest-reset") == "consume-first"


def test_old_exec_keys_still_count():
    s = Settings({"exec.strategy": "rotate"})
    assert s.get("failover.strategy") == "rotate"
    assert not s.is_default("failover.strategy")


def test_session_dir_inherits_finished_onboarding(env):
    import json
    from antigravity_swap import agy, paths
    add_accounts(env, "a@x.io", "b@x.io")
    src = env.gdir / "antigravity-cli" / "cache" / "onboarding.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps({"consumerOnboardingComplete": True, "onboardingComplete": True}))
    sdir = agy.prepare_session(1, default_gdir=env.gdir)
    got = json.loads((sdir / "antigravity-cli" / "cache" / "onboarding.json").read_text())
    assert got["onboardingComplete"] is True


def test_single_login_platform_switches_default_login(env, capfd):
    """Windows-style: no per-account directories; the hop switches the one global login."""
    from antigravity_swap.credentials import FileBackend

    class GlobalBackend(FileBackend):
        isolates_sessions = False

    env.plan({"a@x.io": "quota", "b@x.io": "ok"})
    add_accounts(env, "a@x.io", "b@x.io")
    env.login("a@x.io")
    mgr = env.manager()
    mgr.backend = GlobalBackend(env.gdir)
    err = io.StringIO()
    rc = run_interactive(mgr, [], err=err, grace=0, strategy="rotate")
    out = capfd.readouterr().out
    assert rc == 0, err.getvalue()
    assert "session by b@x.io" in out and "turns=2" in out
    assert env.backend.read().identity().email == "b@x.io"
    assert "default login is now account 2" in err.getvalue()
