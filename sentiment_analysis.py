import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


DEFAULT_SENTIMENT_MODEL = "gpt-5"
DEFAULT_SENTIMENT_MAX_WORKERS = 5

SENTIMENT_VALUES = ["Positive", "Negative", "Neutral"]
SENTIMENT_CATEGORY_VALUES = [
    "Product Performance",
    "Skin Reaction / Tolerance",
    "Ingredients / Formulation",
    "Price / Value",
    "Availability / Retail",
    "Packaging / Usability",
    "Customer Service",
    "Brand Trust / Reputation",
    "Comparison / Alternatives",
    "Routine Fit / Recommendations",
    "Other",
]

SENTIMENT_COLUMNS = [
    "Sentiment",
    "Sentiment_Confidence",
    "Sentiment_Target",
    "Sentiment_Theme",
    "Sentiment_Rationale",
    "Sentiment_Status",
    "Sentiment_Error",
    "Sentiment_Category",
]


def create_openai_client(api_key):
    """
    Creates an OpenAI client using the API key supplied by Streamlit secrets.
    """
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def clean_cell_value(value):
    """
    Converts dataframe values into safe prompt strings.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def build_sentiment_input(row):
    """
    Builds the row-level payload used for target-specific sentiment analysis.
    """
    return {
        "target": clean_cell_value(row.get("Comment_Matched_Terms", "")),
        "comment_body": clean_cell_value(row.get("Comment_Body", "")),
        "post_title": clean_cell_value(row.get("Post_Title", "")),
        "post_body": clean_cell_value(row.get("Post_Body", "")),
    }


def skipped_sentiment_result():
    """
    Returns the standard sentiment fields for rows that cannot be analyzed.
    """
    return {
        "Sentiment": "Neutral",
        "Sentiment_Confidence": 0.0,
        "Sentiment_Target": "",
        "Sentiment_Theme": "",
        "Sentiment_Rationale": "",
        "Sentiment_Status": "skipped",
        "Sentiment_Error": "",
        "Sentiment_Category": "",
    }


def failed_sentiment_result(error):
    """
    Returns the standard sentiment fields for failed OpenAI calls.
    """
    status_code = getattr(error, "status_code", None)
    message = error.__class__.__name__
    if status_code:
        message = f"{message} (status {status_code})"

    return {
        "Sentiment": "Neutral",
        "Sentiment_Confidence": 0.0,
        "Sentiment_Target": "",
        "Sentiment_Theme": "",
        "Sentiment_Rationale": "",
        "Sentiment_Status": "failed",
        "Sentiment_Error": message[:200],
        "Sentiment_Category": "",
    }


def sentiment_response_schema():
    """
    JSON schema for strict OpenAI structured sentiment output.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "sentiment",
            "confidence",
            "target",
            "category",
            "theme",
            "rationale",
        ],
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": SENTIMENT_VALUES,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "target": {
                "type": "string",
            },
            "category": {
                "type": "string",
                "enum": SENTIMENT_CATEGORY_VALUES,
            },
            "theme": {
                "type": "string",
            },
            "rationale": {
                "type": "string",
            },
        },
    }


def extract_response_text(response):
    """
    Extracts text from an OpenAI Responses API response or a test double.
    """
    output_text = getattr(response, "output_text", None)

    if output_text:
        return output_text

    output = getattr(response, "output", None) or []

    for item in output:
        if isinstance(item, dict):
            content = item.get("content") or []
        else:
            content = getattr(item, "content", None) or []

        for content_item in content:
            if isinstance(content_item, dict):
                text = content_item.get("text")
            else:
                text = getattr(content_item, "text", None)

            if text:
                return text

    raise ValueError("OpenAI response did not contain text output.")


def validate_sentiment_payload(payload):
    """
    Validates and normalizes model output into export-ready sentiment fields.
    """
    sentiment = payload["sentiment"]
    confidence = float(payload["confidence"])
    target = str(payload["target"]).strip()
    category = str(payload["category"]).strip()
    theme = str(payload["theme"]).strip()
    rationale = str(payload["rationale"]).strip()

    if sentiment not in SENTIMENT_VALUES:
        raise ValueError("Invalid sentiment value.")

    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("Sentiment confidence must be between 0.0 and 1.0.")

    if category not in SENTIMENT_CATEGORY_VALUES:
        raise ValueError("Invalid sentiment category.")

    if not target or not theme or not rationale:
        raise ValueError("Target, theme, and rationale must not be empty.")

    return {
        "Sentiment": sentiment,
        "Sentiment_Confidence": confidence,
        "Sentiment_Target": target,
        "Sentiment_Category": category,
        "Sentiment_Theme": theme,
        "Sentiment_Rationale": rationale,
        "Sentiment_Status": "analyzed",
        "Sentiment_Error": "",
    }


def _is_transient_error(error):
    """Identify failures that are safe to retry without exposing error text."""
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True

    return error.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _create_response_once(client, request):
    """Call the Responses API, retaining compatibility with clients without store."""
    try:
        return client.responses.create(**request)
    except TypeError as error:
        if "store" not in str(error):
            raise

        fallback_request = dict(request)
        fallback_request.pop("store", None)
        return client.responses.create(**fallback_request)


def create_response(client, model, messages, max_retries=2):
    """
    Calls OpenAI with strict structured output, avoiding retained API data where supported.
    """
    request = {
        "model": model,
        "input": messages,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "target_sentiment",
                "description": "Target-specific sentiment for one Reddit comment.",
                "strict": True,
                "schema": sentiment_response_schema(),
            }
        },
        "store": False,
    }

    for attempt in range(max_retries + 1):
        try:
            return _create_response_once(client, request)
        except Exception as error:
            if not _is_transient_error(error) or attempt >= max_retries:
                raise

            time.sleep(2 ** attempt)


def classify_comment_sentiment(client, row, model=DEFAULT_SENTIMENT_MODEL):
    """
    Classifies one Reddit comment row for sentiment toward its matched target terms.
    """
    sentiment_input = build_sentiment_input(row)

    if not sentiment_input["comment_body"] or not sentiment_input["target"]:
        return skipped_sentiment_result()

    system_prompt = (
        "You classify target-specific sentiment in Reddit comments. "
        "Analyze sentiment only toward the provided target terms. "
        "Ignore sentiment toward other brands, products, or entities. "
        "If the comment is mixed, classify only the sentiment toward the target. "
        "Assign exactly one category from the supplied category list."
    )
    user_prompt = (
        f"Target terms: {sentiment_input['target']}\n\n"
        f"Comment body, the main text to analyze:\n{sentiment_input['comment_body']}\n\n"
        f"Post title, context only:\n{sentiment_input['post_title']}\n\n"
        f"Post body, context only:\n{sentiment_input['post_body']}\n\n"
        f"Allowed categories: {', '.join(SENTIMENT_CATEGORY_VALUES)}\n\n"
        "Example rule: if the target is CeraVe and the comment says "
        "'CeraVe broke me out but La Roche-Posay was great', sentiment is Negative."
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        },
    ]

    try:
        response = create_response(client, model, messages)
        payload = json.loads(extract_response_text(response))

        return validate_sentiment_payload(payload)
    except Exception as error:
        return failed_sentiment_result(error)


def enrich_dataframe_with_sentiment(
    df,
    client,
    model=DEFAULT_SENTIMENT_MODEL,
    progress_callback=None,
    max_workers=DEFAULT_SENTIMENT_MAX_WORKERS,
):
    """
    Appends sentiment columns to the Reddit results dataframe.
    """
    enriched_df = df.copy()

    if enriched_df.empty:
        for column in SENTIMENT_COLUMNS:
            enriched_df[column] = pd.Series(dtype="object")

        return enriched_df

    sentiment_rows = [None] * len(enriched_df)
    total_rows = len(enriched_df)

    try:
        worker_count = max(1, min(int(max_workers), total_rows))
    except (TypeError, ValueError):
        worker_count = DEFAULT_SENTIMENT_MAX_WORKERS
        worker_count = min(worker_count, total_rows)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(classify_comment_sentiment, client, row, model): position
            for position, (_, row) in enumerate(enriched_df.iterrows())
        }

        completed = 0
        for future in as_completed(futures):
            position = futures[future]
            try:
                sentiment_rows[position] = future.result()
            except Exception as error:
                sentiment_rows[position] = failed_sentiment_result(error)

            completed += 1
            if progress_callback:
                progress_callback(f"Analyzed comment {completed} of {total_rows}")

    sentiment_df = pd.DataFrame(sentiment_rows, index=enriched_df.index)

    for column in SENTIMENT_COLUMNS:
        enriched_df[column] = sentiment_df[column]

    return enriched_df
