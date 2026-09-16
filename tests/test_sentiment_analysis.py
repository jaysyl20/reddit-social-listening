import json
import threading
import time
from types import SimpleNamespace

import pandas as pd

from sentiment_analysis import (
    SENTIMENT_COLUMNS,
    SENTIMENT_CATEGORY_VALUES,
    build_sentiment_input,
    classify_comment_sentiment,
    enrich_dataframe_with_sentiment,
)


class FakeResponses:
    def __init__(self, output_text=None, error=None):
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text=None, error=None):
        self.responses = FakeResponses(output_text=output_text, error=error)


class RateLimitError(Exception):
    status_code = 429


def sentiment_row(**overrides):
    row = {
        "Comment_Body": "The cleanser feels gentle and effective.",
        "Comment_Matched_Terms": "CeraVe",
        "Post_Title": "Daily cleanser recommendations",
        "Post_Body": "Looking for a cleanser for sensitive skin.",
    }
    row.update(overrides)
    return pd.Series(row)


def valid_response(sentiment="Positive"):
    return json.dumps({
        "sentiment": sentiment,
        "confidence": 0.92,
        "target": "CeraVe",
        "category": "Product Performance",
        "theme": "Gentle formula",
        "rationale": "The comment describes the target product as gentle and effective.",
    })


def test_build_sentiment_input_uses_comment_as_main_text_and_post_as_context():
    result = build_sentiment_input(sentiment_row())

    assert result["target"] == "CeraVe"
    assert result["comment_body"] == "The cleanser feels gentle and effective."
    assert result["post_title"] == "Daily cleanser recommendations"
    assert result["post_body"] == "Looking for a cleanser for sensitive skin."


def test_missing_comment_or_target_is_skipped_without_calling_openai():
    client = FakeClient(output_text=valid_response())

    result = classify_comment_sentiment(
        client,
        sentiment_row(Comment_Body=""),
        model="gpt-test",
    )

    assert result["Sentiment_Status"] == "skipped"
    assert result["Sentiment"] == "Neutral"
    assert result["Sentiment_Confidence"] == 0.0
    assert client.responses.calls == []


def test_valid_structured_response_is_normalized():
    client = FakeClient(output_text=valid_response())

    result = classify_comment_sentiment(client, sentiment_row(), model="gpt-test")

    assert result["Sentiment"] == "Positive"
    assert result["Sentiment_Confidence"] == 0.92
    assert result["Sentiment_Status"] == "analyzed"
    assert result["Sentiment_Error"] == ""
    request = client.responses.calls[0]
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["schema"]["properties"]["category"]["enum"] == SENTIMENT_CATEGORY_VALUES


def test_target_prompt_keeps_other_brand_sentiment_out_of_scope():
    client = FakeClient(output_text=valid_response(sentiment="Negative"))
    row = sentiment_row(
        Comment_Body="CeraVe broke me out but La Roche-Posay was great.",
        Comment_Matched_Terms="CeraVe",
    )

    result = classify_comment_sentiment(client, row, model="gpt-test")

    prompt = client.responses.calls[0]["input"][1]["content"][0]["text"]
    assert result["Sentiment"] == "Negative"
    assert "La Roche-Posay" in prompt
    assert "Ignore sentiment toward other brands" in client.responses.calls[0]["input"][0]["content"][0]["text"]


def test_invalid_response_is_marked_failed():
    client = FakeClient(output_text=json.dumps({"sentiment": "Positive"}))

    result = classify_comment_sentiment(client, sentiment_row(), model="gpt-test")

    assert result["Sentiment_Status"] == "failed"
    assert result["Sentiment"] == "Neutral"
    assert result["Sentiment_Confidence"] == 0.0
    assert result["Sentiment_Error"]


def test_invalid_category_is_marked_failed():
    payload = json.loads(valid_response())
    payload["category"] = "Not a real category"
    client = FakeClient(output_text=json.dumps(payload))

    result = classify_comment_sentiment(client, sentiment_row(), model="gpt-test")

    assert result["Sentiment_Status"] == "failed"


def test_openai_error_is_marked_failed_and_keeps_row():
    client = FakeClient(error=RuntimeError("temporary API error"))
    df = pd.DataFrame([sentiment_row().to_dict()])

    enriched = enrich_dataframe_with_sentiment(df, client, model="gpt-test")

    assert list(enriched.columns[: len(df.columns)]) == list(df.columns)
    assert all(column in enriched.columns for column in SENTIMENT_COLUMNS)
    assert enriched.loc[0, "Sentiment_Status"] == "failed"
    assert len(enriched) == 1


def test_enrichment_preserves_raw_columns_and_appends_sentiment_columns():
    client = FakeClient(output_text=valid_response())
    df = pd.DataFrame([sentiment_row().to_dict()])

    enriched = enrich_dataframe_with_sentiment(df, client, model="gpt-test")

    assert list(enriched.columns) == list(df.columns) + SENTIMENT_COLUMNS
    assert enriched.loc[0, "Comment_Body"] == df.loc[0, "Comment_Body"]
    assert enriched.loc[0, "Sentiment_Status"] == "analyzed"


def test_transient_response_failure_is_retried(monkeypatch):
    class FlakyResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("rate limited")
            return SimpleNamespace(output_text=valid_response())

    client = SimpleNamespace(responses=FlakyResponses())
    monkeypatch.setattr("sentiment_analysis.time.sleep", lambda _: None)

    result = classify_comment_sentiment(client, sentiment_row(), model="gpt-test")

    assert result["Sentiment_Status"] == "analyzed"
    assert client.responses.calls == 2


def test_parallel_enrichment_preserves_input_order_and_reports_progress():
    class ConcurrentResponses:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def create(self, **kwargs):
            prompt = kwargs["input"][1]["content"][0]["text"]
            comment = prompt.split("Comment body, the main text to analyze:\n", 1)[1].split("\n\n", 1)[0]
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            payload = json.loads(valid_response())
            payload["rationale"] = comment
            return SimpleNamespace(output_text=json.dumps(payload))

    responses = ConcurrentResponses()
    client = SimpleNamespace(responses=responses)
    df = pd.DataFrame([
        sentiment_row(Comment_Body="first").to_dict(),
        sentiment_row(Comment_Body="second").to_dict(),
        sentiment_row(Comment_Body="third").to_dict(),
    ])
    progress = []

    enriched = enrich_dataframe_with_sentiment(
        df,
        client,
        model="gpt-test",
        progress_callback=progress.append,
        max_workers=3,
    )

    assert responses.max_active > 1
    assert list(enriched["Sentiment_Rationale"]) == ["first", "second", "third"]
    assert progress == [
        "Analyzed comment 1 of 3",
        "Analyzed comment 2 of 3",
        "Analyzed comment 3 of 3",
    ]
