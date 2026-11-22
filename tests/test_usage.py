import urllib.error

import pytest

from antigravity_swap import usage


SUMMARY_3P = {"groups": [{"displayName": "Claude and GPT models", "buckets": [
    {"bucketId": "3p-5h", "window": "5h", "remainingFraction": 1},
    {"bucketId": "3p-weekly", "window": "weekly", "remainingFraction": 0.5},
]}]}
MODEL_QUOTA = {"buckets": [
    {"modelId": "gemini-2.5-pro", "remainingFraction": 0, "resetTime": "2026-11-28T15:00:00Z"},
    {"modelId": "gemini-3-flash", "remainingFraction": 0.75},
    {"modelId": "claude-sonnet", "remainingFraction": 1},
]}


def test_fetch_fills_missing_gemini_with_model_quota(monkeypatch):
    calls = []

    def call(host, path, body, token):
        calls.append((host, path, body, token))
        return MODEL_QUOTA if path.endswith(":retrieveUserQuota") else SUMMARY_3P

    monkeypatch.setattr(usage, "_call", call)
    reading = usage.fetch("test-token", "test-project")
    w = usage.windows(reading)
    assert w["gemini-model:gemini-2.5-pro"]["used"] == 100
    assert w["gemini-model:gemini-2.5-pro"]["resets_at"] is not None
    assert w["gemini-model:gemini-3-flash"]["used"] == 25
    assert w["3p-weekly"]["used"] == 50
    assert "gemini-5h" not in w and "gemini-weekly" not in w
    assert len(w) == 4  # the model endpoint must not replace the 3p summary
    assert usage.binding_used(reading, "gemini") == 100
    assert usage.weekly_reset(reading, "gemini") is None
    assert usage.earliest_recovery(reading, "gemini", 90) == w["gemini-model:gemini-2.5-pro"]["resets_at"]
    assert [c[1] for c in calls] == ["/v1internal:retrieveUserQuotaSummary", "/v1internal:retrieveUserQuota"]
    assert all(c[2] == {"project": "test-project"} for c in calls)


def test_fetch_keeps_existing_gemini_windows(monkeypatch):
    def call(host, path, body, token):
        assert path == "/v1internal:retrieveUserQuotaSummary"
        return {"groups": [{"buckets": [
            {"bucketId": "gemini-5h", "remainingFraction": 0.5},
        ]}]}

    monkeypatch.setattr(usage, "_call", call)
    assert usage.windows(usage.fetch("test-token", "test-project"))["gemini-5h"]["used"] == 50


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, None),
    urllib.error.URLError("offline"),
    TimeoutError("timed out"),
    ValueError("invalid JSON"),
])
def test_optional_model_query_failure_keeps_summary(monkeypatch, error):
    def call(host, path, body, token):
        if path.endswith(":retrieveUserQuota"):
            raise error
        return SUMMARY_3P

    monkeypatch.setattr(usage, "_call", call)
    reading = usage.fetch("test-token", "test-project")
    assert usage.windows(reading)["3p-weekly"]["used"] == 50
    assert usage.binding_used(reading, "gemini") is None


def test_model_quota_missing_fraction_is_not_full_or_exhausted():
    assert usage.normalize_models({"buckets": [
        {"modelId": "gemini-unknown", "resetTime": "2026-11-28T15:00:00Z"},
        {"modelId": "gemini-disabled", "remainingFraction": 1, "disabled": True},
        {"modelId": "tab_flash", "remainingFraction": 1},
    ]}) == []
