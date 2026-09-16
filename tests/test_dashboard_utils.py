import pandas as pd

from dashboard_utils import (
    build_category_summary,
    build_sentiment_pie_chart,
    build_sentiment_status_counts,
    build_sentiment_summary,
    build_theme_summary,
)


def test_summary_counts_only_analyzed_rows():
    df = pd.DataFrame([
        {"Sentiment": "Positive", "Sentiment_Status": "analyzed"},
        {"Sentiment": "Negative", "Sentiment_Status": "analyzed"},
        {"Sentiment": "Neutral", "Sentiment_Status": "analyzed"},
        {"Sentiment": "Neutral", "Sentiment_Status": "skipped"},
        {"Sentiment": "Negative", "Sentiment_Status": "failed"},
    ])

    summary = build_sentiment_summary(df)
    counts = dict(zip(summary["Sentiment"], summary["Mentions"]))
    percentages = dict(zip(summary["Sentiment"], summary["Percent"]))

    assert counts == {"Positive": 1, "Negative": 1, "Neutral": 1}
    assert percentages == {"Positive": 33.3, "Negative": 33.3, "Neutral": 33.3}

    status_counts = build_sentiment_status_counts(df)
    assert status_counts == {"total": 5, "analyzed": 3, "skipped": 1, "failed": 1}


def test_theme_summary_groups_case_and_spacing_variants():
    df = pd.DataFrame([
        {
            "Sentiment": "Positive",
            "Sentiment_Status": "analyzed",
            "Sentiment_Theme": "  Hydration  ",
            "Sentiment_Confidence": 0.8,
            "Sentiment_Rationale": "First rationale",
            "Comment_Body": "First comment",
        },
        {
            "Sentiment": "Positive",
            "Sentiment_Status": "analyzed",
            "Sentiment_Theme": "hydration",
            "Sentiment_Confidence": 0.6,
            "Sentiment_Rationale": "Second rationale",
            "Comment_Body": "Second comment",
        },
        {
            "Sentiment": "Positive",
            "Sentiment_Status": "analyzed",
            "Sentiment_Theme": "",
            "Sentiment_Confidence": 0.9,
            "Sentiment_Rationale": "Missing theme rationale",
            "Comment_Body": "Missing theme comment",
        },
        {
            "Sentiment": "Positive",
            "Sentiment_Status": "skipped",
            "Sentiment_Theme": "shipping",
            "Sentiment_Confidence": 0.0,
            "Sentiment_Rationale": "Skipped",
            "Comment_Body": "Skipped",
        },
    ])

    summary = build_theme_summary(df, "Positive")

    hydration = summary[summary["Theme"] == "Hydration"].iloc[0]
    uncategorized = summary[summary["Theme"] == "Uncategorized"].iloc[0]
    assert hydration["Mentions"] == 2
    assert hydration["Percent"] == 66.7
    assert hydration["Example_Rationale"] == "First rationale"
    assert hydration["Example_Comment"] == "First comment"
    assert uncategorized["Mentions"] == 1


def test_theme_summary_is_limited_to_requested_number_of_rows():
    rows = [
        {
            "Sentiment": "Negative",
            "Sentiment_Status": "analyzed",
            "Sentiment_Theme": f"Theme {index}",
            "Sentiment_Confidence": 0.5,
            "Sentiment_Rationale": f"Rationale {index}",
            "Comment_Body": f"Comment {index}",
        }
        for index in range(12)
    ]

    summary = build_theme_summary(pd.DataFrame(rows), "Negative", limit=10)

    assert len(summary) == 10
    assert summary["Theme"].is_unique
    assert set(summary["Theme"]).issubset({f"Theme {index}" for index in range(12)})


def test_category_summary_uses_fixed_categories_and_excludes_failed_rows():
    df = pd.DataFrame([
        {
            "Sentiment": "Negative",
            "Sentiment_Status": "analyzed",
            "Sentiment_Category": "Product Performance",
            "Sentiment_Confidence": 0.8,
            "Sentiment_Rationale": "Performance was poor.",
            "Comment_Body": "It did not work.",
        },
        {
            "Sentiment": "Negative",
            "Sentiment_Status": "analyzed",
            "Sentiment_Category": "product performance",
            "Sentiment_Confidence": 0.6,
            "Sentiment_Rationale": "The result was disappointing.",
            "Comment_Body": "Not effective.",
        },
        {
            "Sentiment": "Positive",
            "Sentiment_Status": "analyzed",
            "Sentiment_Category": "",
            "Sentiment_Confidence": 0.9,
            "Sentiment_Rationale": "It felt good.",
            "Comment_Body": "Nice product.",
        },
        {
            "Sentiment": "Negative",
            "Sentiment_Status": "failed",
            "Sentiment_Category": "Price / Value",
            "Sentiment_Confidence": 0.0,
            "Sentiment_Rationale": "Failed.",
            "Comment_Body": "Failed.",
        },
    ])

    summary = build_category_summary(df, sentiment="Negative")

    assert list(summary["Category"]) == ["Product Performance"]
    assert summary.loc[0, "Mentions"] == 2
    assert summary.loc[0, "Percent"] == 100.0

    all_categories = build_category_summary(df)
    assert "Other" in set(all_categories["Category"])


def test_pie_chart_serializes_with_exact_sentiment_categories():
    summary = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Mentions": [2, 1, 1],
        "Percent": [50.0, 25.0, 25.0],
    })

    spec = build_sentiment_pie_chart(summary).to_dict()

    assert spec["mark"]["type"] == "arc"
    assert spec["encoding"]["color"]["scale"]["domain"] == [
        "Positive",
        "Negative",
        "Neutral",
    ]
