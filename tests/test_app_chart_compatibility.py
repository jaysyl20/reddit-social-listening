from pathlib import Path


def test_app_does_not_pass_width_to_altair_chart():
    source = Path(__file__).parents[1].joinpath("app.py").read_text()

    assert 'st.altair_chart(build_sentiment_pie_chart(summary_df))' in source
    assert 'st.altair_chart(build_sentiment_pie_chart(summary_df), width=' not in source
