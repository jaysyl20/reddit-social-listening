import streamlit as st

from dashboard_utils import (
    build_category_summary,
    build_sentiment_pie_chart,
    build_sentiment_status_counts,
    build_sentiment_summary,
    build_theme_summary,
)
from export_utils import (
    dataframe_to_excel_bytes,
    export_file_names,
    select_export_dataframe,
)
from reddit_scraper import (
    create_reddit_client,
    run_reddit_scraper,
    dataframe_to_csv_bytes
)
from sentiment_analysis import (
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SENTIMENT_MAX_WORKERS,
    create_openai_client,
    enrich_dataframe_with_sentiment,
)


# ==============================
# 1. Page setup
# ==============================

st.set_page_config(
    page_title="Reddit Social Listening Tool",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Reddit Social Listening Tool")

st.write(
    "Track keywords across selected subreddits and export Reddit comment-level data as a CSV."
)


# ==============================
# 2. Helper functions
# ==============================

def parse_multiline_input(text):
    """
    Converts a multiline text box into a clean list.
    Also supports comma-separated values.
    """
    if not text:
        return []

    items = []

    for line in text.splitlines():
        parts = line.split(",")

        for part in parts:
            value = part.strip()
            if value:
                items.append(value)

    return items


def clean_subreddit_name(name):
    """
    Allows users to enter either SkincareAddiction or r/SkincareAddiction.
    """
    name = name.strip()

    if name.lower().startswith("r/"):
        name = name[2:]

    return name


def get_streamlit_secret(secret_name):
    """
    Safely reads Streamlit secrets if they exist.
    If not, returns an empty string.
    """
    try:
        return st.secrets.get(secret_name, "")
    except Exception:
        return ""


def dedupe_list(items):
    """
    Removes duplicates while preserving order.
    """
    cleaned = []
    seen = set()

    for item in items:
        value = str(item).strip()
        key = value.lower()

        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)

    return cleaned


# ==============================
# 3. Reddit credentials
# ==============================

st.sidebar.header("Reddit API Settings")

saved_client_id = get_streamlit_secret("REDDIT_CLIENT_ID")
saved_client_secret = get_streamlit_secret("REDDIT_CLIENT_SECRET")
saved_user_agent = get_streamlit_secret("REDDIT_USER_AGENT")

if saved_client_id and saved_client_secret and saved_user_agent:
    st.sidebar.success("Using saved Reddit API credentials.")
    client_id = saved_client_id
    client_secret = saved_client_secret
    user_agent = saved_user_agent
else:
    st.sidebar.info("Enter Reddit API credentials for testing.")

    client_id = st.sidebar.text_input(
        "Reddit Client ID",
        type="password"
    )

    client_secret = st.sidebar.text_input(
        "Reddit Client Secret",
        type="password"
    )

    user_agent = st.sidebar.text_input(
        "Reddit User Agent",
        value="windows:social_listening_tool:v1.0 by /u/YOUR_USERNAME"
    )


# ==============================
# 4. OpenAI sentiment settings
# ==============================

st.sidebar.header("OpenAI Sentiment Settings")

openai_api_key = get_streamlit_secret("OPENAI_API_KEY")
saved_sentiment_model = get_streamlit_secret("OPENAI_SENTIMENT_MODEL")

if openai_api_key:
    st.sidebar.success("OpenAI API key loaded from Streamlit secrets.")
else:
    st.sidebar.info(
        "Add OPENAI_API_KEY to Streamlit secrets to enable sentiment analysis."
    )

sentiment_model = st.sidebar.text_input(
    "Sentiment Model",
    value=saved_sentiment_model or DEFAULT_SENTIMENT_MODEL,
)


# ==============================
# 5. User inputs
# ==============================

st.header("Search Setup")

keywords_text = st.text_area(
    "Enter keywords to track",
    value="la roche posay\nlrp\nlarocheposay",
    help="Enter one keyword per line. You can also use commas."
)

default_subreddit_options = [
    "SkincareAddiction",
    "SkincareAddictionUK",
    "EuroSkincare",
    "Skincare_Addiction",
    "30PlusSkinCare",
    "AsianBeauty",
    "beauty",
    "MakeupAddiction",
    "tretinoin",
    "Rosacea",
    "eczema",
    "Acne",
    "AusSkincare",
    "CanSkincare"
]

selected_subreddits = st.multiselect(
    "Choose up to 5 subreddits",
    options=default_subreddit_options,
    default=["SkincareAddictionUK", "EuroSkincare"],
    help="Select up to 5 communities to search."
)

custom_subreddits_text = st.text_area(
    "Optional: add custom subreddits",
    value="",
    help="Enter one subreddit per line, without the r/. These will count toward the 5 subreddit maximum."
)

col1, col2 = st.columns(2)

with col1:
    days_back = st.slider(
        "How many days back should the tool search?",
        min_value=7,
        max_value=365,
        value=90,
        step=7
    )

with col2:
    runtime_mode = st.selectbox(
        "Runtime mode",
        options=["Fast", "Medium", "Deep"],
        index=0,
        help="Fast pulls less data. Deep pulls more data but takes longer."
    )


# ==============================
# 6. Runtime explanation
# ==============================

with st.expander("What do the runtime modes mean?"):
    st.write(
        """
        **Fast**: Smaller search, quickest test run.  
        **Medium**: Balanced search depth.  
        **Deep**: Larger search, more comments, slower runtime.
        """
    )


# ==============================
# 7. Run scraper
# ==============================

run_button = st.button("Run Reddit Search", type="primary")

if run_button:
    keywords = parse_multiline_input(keywords_text)

    custom_subreddits = [
        clean_subreddit_name(sub)
        for sub in parse_multiline_input(custom_subreddits_text)
    ]

    all_subreddits = dedupe_list(selected_subreddits + custom_subreddits)

    if not client_id or not client_secret or not user_agent:
        st.error("Please provide Reddit API credentials.")
        st.stop()

    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    if not all_subreddits:
        st.error("Please choose or enter at least one subreddit.")
        st.stop()

    if len(all_subreddits) > 5:
        st.error("Please choose a maximum of 5 subreddits.")
        st.write("You selected:")
        st.write(all_subreddits)
        st.stop()

    st.subheader("Search Summary")

    st.write("**Keywords:**", keywords)
    st.write("**Subreddits:**", all_subreddits)
    st.write("**Days back:**", days_back)
    st.write("**Runtime mode:**", runtime_mode)

    progress_message = st.empty()

    def update_progress(message):
        progress_message.info(message)

    try:
        st.session_state.pop("reddit_sentiment_df", None)

        reddit = create_reddit_client(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        with st.spinner("Running Reddit search..."):
            df = run_reddit_scraper(
                reddit=reddit,
                keywords=keywords,
                subreddits=all_subreddits,
                days_back=days_back,
                runtime_mode=runtime_mode,
                progress_callback=update_progress
            )

        st.session_state["reddit_results_df"] = df

        st.success(f"Search complete. Collected {len(df)} matching comment rows.")

    except Exception as error:
        st.error("Something went wrong while running the Reddit search.")
        st.exception(error)


# ==============================
# 8. Show, analyze, and download results
# ==============================

if "reddit_results_df" in st.session_state:
    df = st.session_state["reddit_results_df"]
    sentiment_df = st.session_state.get("reddit_sentiment_df")
    export_df = select_export_dataframe(df, sentiment_df)

    st.header("Results Preview")

    if df.empty:
        st.warning("No matching comments were found for this search.")
        st.info("The export is still available and will contain the expected column headers.")
    else:
        st.dataframe(export_df.head(100), width="stretch")

        st.header("Sentiment Analysis")

        if openai_api_key:
            if sentiment_df is None:
                sentiment_button_label = "Run Sentiment Analysis"
            else:
                sentiment_button_label = "Rerun Sentiment Analysis"

            if st.button(sentiment_button_label):
                progress_message = st.empty()

                def update_sentiment_progress(message):
                    progress_message.info(message)

                try:
                    openai_client = create_openai_client(openai_api_key)

                    with st.spinner("Running sentiment analysis..."):
                        sentiment_df = enrich_dataframe_with_sentiment(
                            df,
                            client=openai_client,
                            model=sentiment_model.strip() or DEFAULT_SENTIMENT_MODEL,
                            progress_callback=update_sentiment_progress,
                            max_workers=DEFAULT_SENTIMENT_MAX_WORKERS,
                        )

                    st.session_state["reddit_sentiment_df"] = sentiment_df
                    export_df = sentiment_df
                    st.success(
                        f"Sentiment analysis complete for {len(sentiment_df)} comments."
                    )
                except Exception:
                    st.error("Something went wrong while running sentiment analysis.")
        else:
            st.info("Sentiment analysis is unavailable until OPENAI_API_KEY is configured in Streamlit secrets.")

    if sentiment_df is not None:
        st.header("Sentiment Dashboard")
        status_counts = build_sentiment_status_counts(sentiment_df)
        summary_df = build_sentiment_summary(sentiment_df)

        metric_columns = st.columns(6)
        metric_columns[0].metric("Positive", int(summary_df.loc[summary_df["Sentiment"] == "Positive", "Mentions"].iloc[0]))
        metric_columns[1].metric("Negative", int(summary_df.loc[summary_df["Sentiment"] == "Negative", "Mentions"].iloc[0]))
        metric_columns[2].metric("Neutral", int(summary_df.loc[summary_df["Sentiment"] == "Neutral", "Mentions"].iloc[0]))
        metric_columns[3].metric("Analyzed", status_counts["analyzed"])
        metric_columns[4].metric("Skipped", status_counts["skipped"])
        metric_columns[5].metric("Failed", status_counts["failed"])

        if status_counts["analyzed"] == 0:
            st.warning(
                "Sentiment analysis completed, but no rows were successfully analyzed. "
                "The enriched export remains available."
            )
        else:
            st.altair_chart(build_sentiment_pie_chart(summary_df))

            st.subheader("Category Breakdown")
            st.dataframe(
                build_category_summary(sentiment_df),
                width="stretch",
                hide_index=True,
            )

            theme_columns = st.columns(2)
            with theme_columns[0]:
                st.subheader("Top Negative Themes")
                st.dataframe(
                    build_theme_summary(sentiment_df, "Negative"),
                    width="stretch",
                    hide_index=True,
                )

            with theme_columns[1]:
                st.subheader("Top Positive Themes")
                st.dataframe(
                    build_theme_summary(sentiment_df, "Positive"),
                    width="stretch",
                    hide_index=True,
                )

    st.subheader("Downloads")
    csv_data = dataframe_to_csv_bytes(export_df)
    excel_data = dataframe_to_excel_bytes(export_df)
    file_names = export_file_names(sentiment_df)

    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=file_names["csv"],
            mime="text/csv",
        )

    with download_columns[1]:
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=file_names["excel"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
