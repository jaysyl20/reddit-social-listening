from io import BytesIO

import pandas as pd

from export_utils import dataframe_to_excel_bytes, select_export_dataframe


def test_export_selection_prefers_enriched_dataframe():
    raw_df = pd.DataFrame({"Comment_Body": ["raw"]})
    sentiment_df = pd.DataFrame({"Comment_Body": ["enriched"], "Sentiment": ["Positive"]})

    assert select_export_dataframe(raw_df) is raw_df
    assert select_export_dataframe(raw_df, sentiment_df) is sentiment_df


def test_excel_export_returns_index_free_bytes_for_empty_dataframe():
    df = pd.DataFrame(columns=["Comment_Body", "Post_Title"])

    payload = dataframe_to_excel_bytes(df)
    loaded = pd.read_excel(BytesIO(payload))

    assert isinstance(payload, bytes)
    assert list(loaded.columns) == ["Comment_Body", "Post_Title"]
    assert loaded.empty


def test_excel_export_preserves_non_empty_columns_without_index():
    df = pd.DataFrame({"Comment_Body": ["hello"], "Sentiment": ["Positive"]})

    loaded = pd.read_excel(BytesIO(dataframe_to_excel_bytes(df)))

    assert list(loaded.columns) == ["Comment_Body", "Sentiment"]
    assert loaded.to_dict(orient="records") == [
        {"Comment_Body": "hello", "Sentiment": "Positive"}
    ]
