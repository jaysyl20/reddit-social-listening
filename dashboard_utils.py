"""Pure dataframe helpers for the sentiment dashboard."""

import altair as alt
import pandas as pd

from sentiment_analysis import SENTIMENT_CATEGORY_VALUES, SENTIMENT_VALUES

THEME_SUMMARY_COLUMNS = [
    "Theme",
    "Mentions",
    "Percent",
    "Average_Confidence",
    "Example_Rationale",
    "Example_Comment",
]
CATEGORY_SUMMARY_COLUMNS = [
    "Category",
    "Sentiment",
    "Mentions",
    "Percent",
    "Average_Confidence",
    "Example_Rationale",
    "Example_Comment",
]


def _clean_text(value):
    """Return a trimmed string while treating missing dataframe values as blank."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _analyzed_rows(df):
    """Return rows that completed sentiment analysis."""
    if "Sentiment_Status" not in df.columns:
        return df.iloc[0:0].copy()

    return df[
        df["Sentiment_Status"].fillna("").astype(str).str.strip().str.lower()
        == "analyzed"
    ].copy()


def _column_as_series(df, column, default=""):
    """Return a dataframe column or a same-index fallback series."""
    if column in df.columns:
        return df[column]

    return pd.Series(default, index=df.index)


def build_sentiment_summary(df):
    """Build counts and percentages for analyzed Positive/Negative/Neutral rows."""
    analyzed_df = _analyzed_rows(df)
    sentiment_series = _column_as_series(analyzed_df, "Sentiment")
    sentiment_series = sentiment_series.fillna("").astype(str).str.strip()
    counts = sentiment_series.value_counts()
    total = int(len(analyzed_df))

    rows = []
    for sentiment in SENTIMENT_VALUES:
        mentions = int(counts.get(sentiment, 0))
        percent = (mentions / total * 100) if total else 0.0
        rows.append({
            "Sentiment": sentiment,
            "Mentions": mentions,
            "Percent": round(percent, 1),
        })

    return pd.DataFrame(rows, columns=["Sentiment", "Mentions", "Percent"])


def build_sentiment_status_counts(df):
    """Return analyzed, skipped, failed, and total row counts for diagnostics."""
    statuses = (
        df["Sentiment_Status"].fillna("").astype(str).str.strip().str.lower()
        if "Sentiment_Status" in df.columns
        else pd.Series(dtype="object")
    )

    return {
        "total": int(len(df)),
        "analyzed": int((statuses == "analyzed").sum()),
        "skipped": int((statuses == "skipped").sum()),
        "failed": int((statuses == "failed").sum()),
    }


def build_theme_summary(df, sentiment, limit=10):
    """Group analyzed rows by normalized theme and retain representative text."""
    analyzed_df = _analyzed_rows(df)
    if "Sentiment" not in analyzed_df.columns:
        return pd.DataFrame(columns=THEME_SUMMARY_COLUMNS)

    target_df = analyzed_df[
        analyzed_df["Sentiment"].fillna("").astype(str).str.strip() == sentiment
    ].copy()
    if target_df.empty:
        return pd.DataFrame(columns=THEME_SUMMARY_COLUMNS)

    target_df["Theme"] = _column_as_series(
        target_df, "Sentiment_Theme"
    ).map(_clean_text)
    target_df.loc[target_df["Theme"] == "", "Theme"] = "Uncategorized"
    target_df["_Theme_Key"] = target_df["Theme"].str.casefold()
    target_df["_Confidence"] = pd.to_numeric(
        _column_as_series(target_df, "Sentiment_Confidence", 0.0),
        errors="coerce",
    ).fillna(0.0)
    target_df["_Rationale"] = _column_as_series(
        target_df, "Sentiment_Rationale"
    ).map(_clean_text)
    target_df["_Comment"] = _column_as_series(
        target_df, "Comment_Body"
    ).map(_clean_text)

    rows = []
    for _, group in target_df.groupby("_Theme_Key", sort=False):
        representative = group.iloc[0]
        rows.append({
            "Theme": representative["Theme"],
            "Mentions": int(len(group)),
            "Percent": round(len(group) / len(target_df) * 100, 1),
            "Average_Confidence": round(float(group["_Confidence"].mean()), 3),
            "Example_Rationale": representative["_Rationale"],
            "Example_Comment": representative["_Comment"],
        })

    summary_df = pd.DataFrame(rows, columns=THEME_SUMMARY_COLUMNS)
    if summary_df.empty:
        return summary_df

    return (
        summary_df.sort_values(
            ["Mentions", "Average_Confidence", "Theme"],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(limit)
        .reset_index(drop=True)
    )


def build_category_summary(df, sentiment=None, limit=None):
    """Group analyzed rows by the fixed category taxonomy."""
    analyzed_df = _analyzed_rows(df)
    if analyzed_df.empty:
        return pd.DataFrame(columns=CATEGORY_SUMMARY_COLUMNS)

    if sentiment is not None:
        analyzed_df = analyzed_df[
            analyzed_df["Sentiment"].fillna("").astype(str).str.strip() == sentiment
        ].copy()

    if analyzed_df.empty:
        return pd.DataFrame(columns=CATEGORY_SUMMARY_COLUMNS)

    analyzed_df["Category"] = _column_as_series(
        analyzed_df, "Sentiment_Category"
    ).map(_clean_text)
    known_categories = {
        category.casefold(): category for category in SENTIMENT_CATEGORY_VALUES
    }
    analyzed_df["Category"] = analyzed_df["Category"].map(
        lambda value: known_categories.get(value.casefold(), "Other")
        if value
        else "Other"
    )
    analyzed_df["_Category_Key"] = analyzed_df["Category"].str.casefold()
    analyzed_df["_Sentiment"] = _column_as_series(
        analyzed_df, "Sentiment"
    ).map(_clean_text)
    analyzed_df["_Confidence"] = pd.to_numeric(
        _column_as_series(analyzed_df, "Sentiment_Confidence", 0.0),
        errors="coerce",
    ).fillna(0.0)
    analyzed_df["_Rationale"] = _column_as_series(
        analyzed_df, "Sentiment_Rationale"
    ).map(_clean_text)
    analyzed_df["_Comment"] = _column_as_series(
        analyzed_df, "Comment_Body"
    ).map(_clean_text)

    rows = []
    group_columns = ["_Category_Key"] if sentiment is not None else [
        "_Category_Key",
        "_Sentiment",
    ]
    for _, group in analyzed_df.groupby(group_columns, sort=False):
        representative = group.iloc[0]
        rows.append({
            "Category": representative["Category"],
            "Sentiment": representative["_Sentiment"],
            "Mentions": int(len(group)),
            "Percent": round(len(group) / len(analyzed_df) * 100, 1),
            "Average_Confidence": round(float(group["_Confidence"].mean()), 3),
            "Example_Rationale": representative["_Rationale"],
            "Example_Comment": representative["_Comment"],
        })

    summary_df = pd.DataFrame(rows, columns=CATEGORY_SUMMARY_COLUMNS)
    summary_df = summary_df.sort_values(
        ["Mentions", "Average_Confidence", "Category", "Sentiment"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    if limit is not None:
        summary_df = summary_df.head(limit).reset_index(drop=True)

    return summary_df


def build_sentiment_pie_chart(summary_df):
    """Create a fixed-category Altair pie chart from a sentiment summary."""
    chart_data = summary_df.copy()
    chart_data["Sentiment"] = chart_data["Sentiment"].astype(str)

    return (
        alt.Chart(chart_data)
        .mark_arc()
        .encode(
            theta=alt.Theta("Mentions:Q", stack=True),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=SENTIMENT_VALUES,
                    range=["#2ca02c", "#d62728", "#7f8c8d"],
                ),
                legend=alt.Legend(title="Sentiment"),
            ),
            tooltip=[
                alt.Tooltip("Sentiment:N", title="Sentiment"),
                alt.Tooltip("Mentions:Q", title="Mentions"),
                alt.Tooltip("Percent:Q", title="Percent", format=".1f"),
            ],
        )
        .properties(width="container", height=360)
    )
