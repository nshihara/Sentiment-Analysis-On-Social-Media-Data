import streamlit as st
import tweepy
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import io

# Download necessary NLTK data
nltk.download('vader_lexicon')

# --- Twitter API Authentication ---
# Replace 'YOUR_BEARER_TOKEN' with your actual Twitter API Bearer Token
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAE%2BGxAEAAAAAZfl%2BLhLifBkNy1fiyUEe%2F2xCldE%3DNrWiq1MgPhLV6RTIXlgtVX4evpUUatsGlJfZxQr5shsFupH7ue'
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

# --- Function to Fetch Tweets ---
def fetch_tweets(query, max_results=100, start_time=None, end_time=None, retries=5, retry_wait=15 * 60):
    """
    Fetch tweets using Twitter API with automatic retries on rate limits or errors.
    
    Parameters:
        query (str): The search query.
        max_results (int): Number of tweets to fetch (10-100).
        start_time (str): ISO 8601 format start time.
        end_time (str): ISO 8601 format end time.
        retries (int): Number of retry attempts.
        retry_wait (int): Wait time between retries in seconds.
    
    Returns:
        list: List of tweet texts.
    """
    for attempt in range(retries):
        try:
            response = client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at']
            )
            if response.data:
                return [tweet.text for tweet in response.data]
            else:
                return []
        except tweepy.errors.TooManyRequests:
            if attempt < retries - 1:
                st.warning(f"Rate limit reached. Retrying in {retry_wait // 60} minutes...")
                time.sleep(retry_wait)
            else:
                st.error("Maximum retries reached. Please try again later.")
                return []
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return []

# --- Function to Preprocess Tweets ---
def preprocess_tweet(sen):
    """
    Clean tweet text by removing URLs, mentions, special characters, and extra spaces.
    
    Parameters:
        sen (str): The tweet text.
    
    Returns:
        str: Cleaned tweet text.
    """
    sentence = sen.lower()
    sentence = re.sub('RT @\w+: ', " ", sentence)
    sentence = re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

# --- Function to Perform Sentiment Analysis ---
def analyze_sentiments(df):
    """
    Analyze sentiment of tweets using TextBlob and VADER.
    
    Parameters:
        df (DataFrame): DataFrame containing tweet texts.
    
    Returns:
        DataFrame: DataFrame with added sentiment analysis columns.
    """
    df['cleaned'] = df['text'].apply(preprocess_tweet)
    df[['polarity', 'subjectivity']] = df['cleaned'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    sid = SentimentIntensityAnalyzer()
    
    sentiments = []
    neg = []
    neu = []
    pos = []
    compound = []
    
    for text in df['cleaned']:
        score = sid.polarity_scores(text)
        comp = score['compound']
        if comp > 0.05:
            sentiments.append("positive")
        elif comp < -0.05:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
        neg.append(score['neg'])
        neu.append(score['neu'])
        pos.append(score['pos'])
        compound.append(comp)
    
    df['sentiment'] = sentiments
    df['neg'] = neg
    df['neu'] = neu
    df['pos'] = pos
    df['compound'] = compound
    return df

# --- Function to Plot Donut Chart ---
def plot_donut_chart(df, column):
    """
    Plot a donut chart showing the distribution of sentiments.
    
    Parameters:
        df (DataFrame): DataFrame containing sentiment data.
        column (str): Column name for sentiments.
    """
    sentiment_counts = df[column].value_counts(normalize=True) * 100
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    colors = ['green', 'blue', 'red']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# --- Function to Create Word Cloud ---
def create_wordcloud(text):
    """
    Generate and display a word cloud from the provided text.
    
    Parameters:
        text (array-like): List or array of text data.
    """
    wc = WordCloud(
        background_color="white",
        max_words=100,
        stopwords=set(STOPWORDS),
        repeat=True
    )
    wc.generate(' '.join(text))
    fig, ax = plt.subplots(figsize=(10,7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# --- Streamlit Interface ---
def main():
    st.markdown(
        """
        <h1 style="color: #1DA1F2; text-align: center;">
            📈 Twitter Sentiment Analysis
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    This application fetches recent tweets based on your query, performs sentiment analysis, 
    visualizes the results, and allows you to download the analysis data.
    """)
    
    # --- Sidebar for Inputs ---
    st.sidebar.header("Query Settings")
    
    # Toggle for date range
    use_date = st.sidebar.checkbox("Use specific date range for query", value=False)
    
    if use_date:
        # Define date range inputs
        today = datetime.utcnow().date()
        default_start = today - timedelta(days=7)
        start_date = st.sidebar.date_input("Start date", default_start)
        end_date = st.sidebar.date_input("End date", today)
        
        if start_date > end_date:
            st.sidebar.error("Start date must be before end date.")
            st.stop()
        
        # Convert to ISO 8601 format
        start_time = datetime.combine(start_date, datetime.min.time()).isoformat("T") + "Z"
        end_time_iso = datetime.combine(end_date, datetime.max.time()).isoformat("T") + "Z"
    else:
        start_time = None
        end_time_iso = None
    
    # --- Language Selection ---
    language_options = {
        "Any": "",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh",
        "Russian": "ru",
        "Arabic": "ar",
        # Add more languages as needed
    }
    
    selected_language = st.sidebar.selectbox(
        "Select Language",
        options=list(language_options.keys()),
        index=0
    )
    lang_code = language_options[selected_language]
    
    # --- Retweet Toggle ---
    exclude_retweets = st.sidebar.checkbox("Exclude Retweets", value=True)
    
    # --- Keywords and Hashtags Input ---
    st.sidebar.subheader("Search Keywords & Hashtags")
    
    keywords_input = st.sidebar.text_input(
        "Keywords",
        value="",
        help="Enter keywords separated by spaces. Do not include '#' symbol."
    )
    
    hashtags_input = st.sidebar.text_input(
        "Hashtags",
        value="",
        help="Enter hashtags separated by commas or spaces. Do not include '#' symbol."
    )
    
    # --- Max results ---
    max_results = st.sidebar.slider(
        "Number of tweets to fetch",
        min_value=10,
        max_value=100,
        value=100,
        step=10,
        help="Select the number of tweets to fetch (10-100)"
    )
    
    # --- Fetch Tweets Button ---
    if st.sidebar.button("Fetch Tweets"):
        # Build the query based on user inputs
        query_parts = []
        
        # Process keywords
        if keywords_input:
            # Split keywords by spaces and join with OR
            keywords = keywords_input.strip().split()
            keywords_query = ' OR '.join(keywords)
            query_parts.append(f"({keywords_query})")
        
        # Process hashtags
        if hashtags_input:
            # Split hashtags by commas or spaces
            hashtags = re.split('[, ]+', hashtags_input.strip())
            hashtags = [f"#{tag}" for tag in hashtags if tag]  # Add '#' prefix
            if hashtags:
                hashtags_query = ' OR '.join(hashtags)
                query_parts.append(f"({hashtags_query})")
        
        # Combine query parts
        query = ' '.join(query_parts)
        
        # Add language filter
        if lang_code:
            query += f" lang:{lang_code}"
        
        # Add retweet filter
        if exclude_retweets:
            query += " -is:retweet"
        
        if not query.strip():
            st.warning("Please enter at least one keyword or hashtag for the query.")
        else:
            with st.spinner("Fetching tweets..."):
                tweets = fetch_tweets(
                    query=query,
                    max_results=max_results,
                    start_time=start_time,
                    end_time=end_time_iso
                )
            
            if not tweets:
                st.warning("No tweets found for the given query and date range.")
            else:
                # Create DataFrame
                df = pd.DataFrame(tweets, columns=["text"])
                
                # Perform Sentiment Analysis
                df = analyze_sentiments(df)
                
                # Display DataFrame
                st.subheader("Fetched Tweets")
                st.dataframe(df[['text', 'sentiment', 'polarity', 'subjectivity', 'compound']].head(10))
                st.markdown(f"... showing 10 of {len(df)} tweets.")
                
                # --- Sentiment Distribution ---
                st.subheader("📊 Sentiment Distribution")
                plot_donut_chart(df, "sentiment")
                
                # --- Word Cloud ---
                st.subheader("☁️ Word Cloud")
                create_wordcloud(df['cleaned'].values)
                
                # --- Additional Insights ---
                df['text_len'] = df['cleaned'].str.len()
                df['text_word_count'] = df['cleaned'].apply(lambda x: len(str(x).split()))
                
                st.subheader("📈 Additional Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Average Text Length by Sentiment:**")
                    avg_length = df.groupby("sentiment")['text_len'].mean().round(2)
                    st.bar_chart(avg_length)
                
                with col2:
                    st.markdown("**Average Word Count by Sentiment:**")
                    avg_word_count = df.groupby("sentiment")['text_word_count'].mean().round(2)
                    st.bar_chart(avg_word_count)
                
                # --- Download Button ---
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Sentiment Analysis as CSV",
                    data=csv,
                    file_name='sentiment_analysis_output.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
