import tweepy
import streamlit as st
import pandas as pd
from transformers import pipeline
import configparser
from textblob import TextBlob
import re
import seaborn as sns
import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


classifier = pipeline('sentiment-analysis')


def run():
    st.title("The Brandor Web App 🔥")
    st.subheader('Twitter Monitoring Tool for Customer Feedback using Sentiment Analysis')
    st.markdown('This app performs sentiment analysis on tweets from Twitter by entering the handle of business '
                'account (or any account at all). The "Tweet Generator" is a feature that streams the most recent '
                'tweets from the designated account and displays them in a Dataframe table. The "Tweet Analyzer" '
                'performs sentiment analysis on these tweets, returning their subjectivity, polarity, and ultimately, '
                'sentiment. The Brandor Tool also has a "Data Visualizer" feature that lets users see the tweet '
                'analysis in form of a bar chart.')

    menu = ["Sign Up", "Log In", "Tweet Analyzer", "Tweet Generator"]
    choice = st.sidebar.selectbox("Select Action", menu)

    if choice == "Log In":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.button("Log In"):
            create_usertable()
            result = login_user(username, password)
            if result:
                st.success("Logged in as {}".format(username))
                st.info("Go to Tweet Analyzer or Tweet Generator to analyze or generate tweets.")
            else:
                st.warning("Incorrect username or password. Try again.")

    elif choice == "Sign Up":
        st.subheader("Create account")
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.button("Sign up"):
            create_usertable()
            add_userdata(new_user, new_password)
            st.success("Welcome to Brandor. Thank you for doing business with us.")
            st.info("Go to menu to log in to your new Brandor account.")

    elif choice == "Tweet Analyzer":
        st.subheader("Analyze the tweets of your Twitter business")
        st.subheader("This tool performs the following:")
        st.write("1. Extracts the last 100 tweets from the given twitter handle")
        st.write(
            "2. Performs sentiment analysis on the streamed tweets and displays it in form of a bar graph")

        raw_text = st.text_area("Enter the Twitter handle of the business (without the '@')")

        st.markdown("Check out the menu for alternate options")

        analyzer_choice = st.selectbox("Select the Activities",
                                       ["Show Recent Tweets", "Data Visualizer"])

        if st.button("Analyze"):
            if analyzer_choice == "Show Recent Tweets":
                st.success("Fetching last 100 Tweets")

                def Show_Recent_Tweets(raw_text):
                    # Extract 100 tweets from the Twitter user
                    posts = api.user_timeline(screen_name=raw_text, count=100, lang="en",
                                              tweet_mode="extended")

                    def get_tweets():
                        the_list = []
                        i = 1
                        for tweet in posts[:100]:
                            the_list.append(tweet.full_text)
                            i = i + 1
                        return the_list

                    user_recent_tweets = get_tweets()
                    return user_recent_tweets

                recent_tweets = Show_Recent_Tweets(raw_text)
                st.write(recent_tweets)

            else:
                def Plot_Analysis():
                    st.success("Generating Visualisation for Sentiment Analysis")

                    posts = api.user_timeline(screen_name=raw_text, count=100, lang="en",
                                              tweet_mode="extended")
                    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

                    # Create a function to clean the tweets
                    def cleanTxt(text):
                        text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                        text = re.sub('#', '', text)  # Removing '#' hash tag
                        text = re.sub('RT[\s]+', '', text)  # Removing RT
                        text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

                        return text

                    # Clean the tweets
                    df['Tweets'] = df['Tweets'].apply(cleanTxt)

                    def getSubjectivity(text):
                        return TextBlob(text).sentiment.subjectivity

                    # Create a function to get the polarity
                    def getPolarity(text):
                        return TextBlob(text).sentiment.polarity

                    # Create two new columns 'Subjectivity' & 'Polarity'
                    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
                    df['Polarity'] = df['Tweets'].apply(getPolarity)

                    def getAnalysis(score):
                        if score < 0:
                            return 'Negative'
                        elif score == 0:
                            return 'Neutral'
                        else:
                            return 'Positive'

                    df['Analysis'] = df['Polarity'].apply(getAnalysis)

                    return df

                df = Plot_Analysis()

                st.write(sns.countplot(x=df["Analysis"], data=df))

                st.pyplot(use_container_width=True)
                st.set_option('deprecation.showPyplotGlobalUse', False)

    else:
        st.subheader("This tool fetches the last 100 tweets from the Twitter handle & performs the "
                     "following tasks")

        st.write("1. Converts it into a DataFrame")
        st.write("2. Cleans the original tweet")
        st.write("3. Analyzes subjectivity of tweets and adds a column for it")
        st.write("4. Analyzes polarity of tweets and adds a column for it")
        st.write("5. Analyzes sentiments of tweets and adds a column for it")

        user_name = st.text_area("*Enter the exact twitter handle of the business account (without @)*")

        st.markdown("Check out the sidebar for alternate options")

        def get_data(account_user_name):

            posts = api.user_timeline(screen_name=account_user_name, count=100, lang="en", tweet_mode="extended")
            df = pd.DataFrame([tweet.full_text for tweet in posts],
                              columns=['Tweets'])

            def cleanTxt(text):
                text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                text = re.sub('#', '', text)  # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text)  # Removing RT
                text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                return text

            # Clean the tweets
            df['Tweets'] = df['Tweets'].apply(cleanTxt)

            def getSubjectivity(text):
                return TextBlob(text).sentiment.subjectivity

            def getPolarity(text):
                return TextBlob(text).sentiment.polarity

            # def getCreatedAt(text):
            #     return text.created_at
            #
            # def getLocation(text):
            #     return text.location

            # Create two new columns 'Subjectivity' & 'Polarity'
            df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
            df['Polarity'] = df['Tweets'].apply(getPolarity)
            # df['Created at'] = df['Tweets'].apply(getCreatedAt)
            # df['Location'] = df['Tweets'].apply(getLocation)

            def getAnalysis(score):
                if score < 0:
                    return 'Negative'
                elif score == 0:
                    return 'Neutral'
                else:
                    return 'Positive'

            df['Analysis'] = df['Polarity'].apply(getAnalysis)
            return df

        if st.button("Show Data"):
            st.success("Streaming 100 recent tweets")

            df = get_data(user_name)

            st.write(df)

    st.subheader('Author :  IBIAM CHRISTOPHER :sunglasses:')


if __name__ == '__main__':
    run()
