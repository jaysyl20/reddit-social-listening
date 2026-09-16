"""Export and export-selection helpers for Reddit result dataframes."""

from io import BytesIO

import pandas as pd


def select_export_dataframe(raw_df, sentiment_df=None):
    """Use enriched results when available, otherwise use raw Reddit results."""
    return sentiment_df if sentiment_df is not None else raw_df


def dataframe_to_excel_bytes(df):
    """Convert a dataframe to an index-free XLSX byte payload."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Reddit Results")

    return output.getvalue()


def export_file_names(sentiment_df=None):
    """Return download filenames that identify raw or sentiment-enriched data."""
    prefix = "reddit_social_listening_sentiment" if sentiment_df is not None else "reddit_social_listening"
    return {
        "csv": f"{prefix}_export.csv",
        "excel": f"{prefix}_export.xlsx",
    }
