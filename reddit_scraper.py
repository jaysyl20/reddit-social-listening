import praw
import pandas as pd
import unicodedata
import re
import time
from datetime import datetime, timezone, timedelta


# ==============================
# 1. Reddit client helper
# ==============================

def create_reddit_client(client_id, client_secret, user_agent):
    """
    Creates and returns a Reddit API client.
    """
    return praw.Reddit(
    	client_id='YlONgHYYAgjib1xg0Ddwzg',
    	client_secret='0vtLqCsqtjm8IMmcw7-1IefJrf7M_g',
    	user_agent='windows:social_listening_tool:v1.0 (by /u/JayPublicis)'
    )


# ==============================
# 2. Text cleaning helpers
# ==============================

def normalize_text(text):
    """
    Lowercases text, removes accents, removes punctuation,
    and standardises spacing.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFD", str(text))
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_list(items, max_items=None):
    """
    Cleans a list of user inputs:
    - removes blank values
    - strips spaces
    - removes duplicate values
    - optionally limits the number of items
    """
    cleaned = []
    seen = set()

    for item in items:
        value = str(item).strip()

        if not value:
            continue

        # Allow users to type either SkincareAddiction or r/SkincareAddiction
        if value.lower().startswith("r/"):
            value = value[2:]

        key = value.lower()

        if key not in seen:
            cleaned.append(value)
            seen.add(key)

    if max_items:
        cleaned = cleaned[:max_items]

    return cleaned


def build_search_query(keyword):
    """
    Wraps multi-word keywords in quotes for Reddit search.
    Example:
    la roche posay -> "la roche posay"
    lrp -> lrp
    """
    keyword = str(keyword).strip()

    if " " in keyword:
        return f'"{keyword}"'

    return keyword


def find_matched_terms(text, keywords):
    """
    Returns keyword terms found in the text.
    Uses word boundaries so short terms like 'lrp' only match as standalone terms.
    """
    normalized_text = normalize_text(text)

    normalized_keywords = sorted(
        list(set(normalize_text(keyword) for keyword in keywords)),
        key=len,
        reverse=True
    )

    matches = []

    for term in normalized_keywords:
        if not term:
            continue

        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, normalized_text):
            matches.append(term)

    return ", ".join(matches)


def query_mentioned_in_text(search_query, text):
    """
    Checks whether the specific search query appears in the text.
    """
    clean_query = normalize_text(search_query.replace('"', ""))
    clean_text = normalize_text(text)

    if not clean_query or not clean_text:
        return "No"

    pattern = r"\b" + re.escape(clean_query) + r"\b"

    return "Yes" if re.search(pattern, clean_text) else "No"


def utc_datetime_string(timestamp):
    """
    Converts Reddit UTC timestamp into readable date/time.
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def reddit_post_url(post):
    """
    Builds a full Reddit post URL.
    """
    return "https://www.reddit.com" + post.permalink


def reddit_comment_url(comment):
    """
    Builds a full Reddit comment URL.
    """
    return "https://www.reddit.com" + comment.permalink


# ==============================
# 3. Runtime settings
# ==============================

def get_runtime_settings(runtime_mode):
    """
    Converts runtime mode into scraper limits.

    Fast = quicker run, smaller dataset
    Medium = balanced
    Deep = slower run, larger dataset
    """

    runtime_mode = str(runtime_mode).strip().lower()

    settings = {
        "fast": {
            "post_search_limit_per_query": 50,
            "max_posts_to_check_per_subreddit": 50,
            "comment_more_limit": 0,
            "request_pause_seconds": 0
        },
        "medium": {
            "post_search_limit_per_query": 100,
            "max_posts_to_check_per_subreddit": 150,
            "comment_more_limit": 1,
            "request_pause_seconds": 0.1
        },
        "deep": {
            "post_search_limit_per_query": 200,
            "max_posts_to_check_per_subreddit": 300,
            "comment_more_limit": 2,
            "request_pause_seconds": 0.15
        }
    }

    return settings.get(runtime_mode, settings["fast"])


# ==============================
# 4. Main scraper function
# ==============================

def run_reddit_scraper(
    reddit,
    keywords,
    subreddits,
    days_back=90,
    runtime_mode="Fast",
    progress_callback=None
):
    """
    Runs Reddit scraping and returns a comment-level dataframe.

    Inputs:
    - reddit: authenticated PRAW Reddit client
    - keywords: list of keywords the user wants to track
    - subreddits: list of subreddits, max 5
    - days_back: number of days back to search
    - runtime_mode: Fast, Medium, or Deep
    - progress_callback: optional function for front-end progress updates

    Output:
    - pandas DataFrame
    """

    keywords = clean_list(keywords)
    subreddits = clean_list(subreddits, max_items=5)
    settings = get_runtime_settings(runtime_mode)

    if not keywords:
        raise ValueError("Please provide at least one keyword.")

    if not subreddits:
        raise ValueError("Please provide at least one subreddit.")

    search_queries = [build_search_query(keyword) for keyword in keywords]

    cutoff_datetime = datetime.now(timezone.utc) - timedelta(days=days_back)
    cutoff_timestamp = cutoff_datetime.timestamp()

    comment_rows = []
    seen_post_ids = set()
    seen_comment_ids = set()

    def report(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)

    report("Starting Reddit collection...")

    for sub in subreddits:
        report(f"Searching subreddit: r/{sub}")

        subreddit = reddit.subreddit(sub)
        posts_checked_for_subreddit = 0

        for search_query in search_queries:
            report(f"Query: {search_query}")

            try:
                search_results = subreddit.search(
                    search_query,
                    sort="new",
                    time_filter="year",
                    limit=settings["post_search_limit_per_query"]
                )

                for post in search_results:
                    if posts_checked_for_subreddit >= settings["max_posts_to_check_per_subreddit"]:
                        report(f"Reached post check limit for r/{sub}")
                        break

                    # Keep only posts within the selected date range
                    if post.created_utc < cutoff_timestamp:
                        continue

                    # Avoid checking the same post more than once
                    if post.id in seen_post_ids:
                        continue

                    seen_post_ids.add(post.id)
                    posts_checked_for_subreddit += 1

                    post_title = post.title or ""
                    post_body = post.selftext or ""
                    post_date = utc_datetime_string(post.created_utc)
                    post_url = reddit_post_url(post)

                    search_query_in_title = query_mentioned_in_text(search_query, post_title)
                    search_query_in_body = query_mentioned_in_text(search_query, post_body)

                    title_matched_terms = find_matched_terms(post_title, keywords)
                    body_matched_terms = find_matched_terms(post_body, keywords)

                    try:
                        post.comment_sort = "top"
                        post.comments.replace_more(
                            limit=settings["comment_more_limit"]
                        )

                        for comment in post.comments.list():
                            comment_body = getattr(comment, "body", "") or ""

                            comment_matched_terms = find_matched_terms(
                                comment_body,
                                keywords
                            )

                            # Only keep comments that mention one or more tracked keywords
                            if not comment_matched_terms:
                                continue

                            if comment.id in seen_comment_ids:
                                continue

                            seen_comment_ids.add(comment.id)

                            comment_rows.append({
                                "Subreddit": sub,
                                "Search_Query": search_query,

                                "Post_ID": post.id,
                                "Post_Date": post_date,
                                "Post_Title": post_title,
                                "Post_Body": post_body,
                                "Post_URL": post_url,
                                "Post_Score": post.score,
                                "Post_Num_Comments": post.num_comments,

                                "Search_Query_In_Post_Title": search_query_in_title,
                                "Search_Query_In_Post_Body": search_query_in_body,
                                "Title_Matched_Terms": title_matched_terms,
                                "Body_Matched_Terms": body_matched_terms,

                                "Comment_ID": comment.id,
                                "Comment_Date": utc_datetime_string(comment.created_utc),
                                "Comment_Body": comment_body,
                                "Comment_Score": comment.score,
                                "Comment_URL": reddit_comment_url(comment),
                                "Comment_Matched_Terms": comment_matched_terms
                            })

                    except Exception as comment_error:
                        report(
                            f"Could not collect comments for post {post.id}: {comment_error}"
                        )

                    time.sleep(settings["request_pause_seconds"])

            except Exception as search_error:
                report(
                    f"Search failed for r/{sub}, query {search_query}: {search_error}"
                )

            if posts_checked_for_subreddit >= settings["max_posts_to_check_per_subreddit"]:
                break

    comments_df = pd.DataFrame(comment_rows, columns=[
        "Subreddit",
        "Search_Query",

        "Post_ID",
        "Post_Date",
        "Post_Title",
        "Post_Body",
        "Post_URL",
        "Post_Score",
        "Post_Num_Comments",

        "Search_Query_In_Post_Title",
        "Search_Query_In_Post_Body",
        "Title_Matched_Terms",
        "Body_Matched_Terms",

        "Comment_ID",
        "Comment_Date",
        "Comment_Body",
        "Comment_Score",
        "Comment_URL",
        "Comment_Matched_Terms"
    ])

    report(f"Collection complete. Total matching comments collected: {len(comments_df)}")

    return comments_df


# ==============================
# 5. CSV export helper
# ==============================

def dataframe_to_csv_bytes(df):
    """
    Converts dataframe to CSV bytes for Streamlit download.
    """
    return df.to_csv(index=False).encode("utf-8")