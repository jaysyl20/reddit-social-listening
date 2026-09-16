# reddit-social-listening

Streamlit Reddit keyword tracking tool with optional target-specific sentiment analysis.

## Setup

Install the application dependencies:

```bash
python -m pip install -r requirements.txt
```

For local tests, install the development dependency:

```bash
python -m pip install -r requirements-dev.txt
```

## Run

Start the app with:

```bash
streamlit run app.py
```

Enter Reddit API credentials in the sidebar, then choose keywords and up to five
subreddits before running the Reddit search. The raw results remain available as
a CSV download, and can also be downloaded as Excel.

After results are collected, configure the OpenAI key in
`.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_SENTIMENT_MODEL = "gpt-5"
```

The OpenAI key is used only when the user clicks **Run Sentiment Analysis**.
Sentiment results are added to the export and drive the dashboard pie chart and
theme summaries.
