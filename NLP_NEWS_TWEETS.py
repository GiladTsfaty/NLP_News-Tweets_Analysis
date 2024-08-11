#####  imports  #####
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import spacy
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer



# Import necessary modules for RNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Download necessary NLTK data
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')



# Load spaCy model
nlp = spacy.load("en_core_web_sm")










# #####  DATA EXTRACTION  #####
#
# # Load the datasets with low_memory=False to avoid dtype warnings
# cnn_tweets = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\tweets_cnn.csv', low_memory=False)
# bbc_tweets = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\tweets_bbc.csv', low_memory=False)
#
# # Drop problematic columns (22 and 24)
# cnn_tweets.drop(cnn_tweets.columns[[22, 24]], axis=1, inplace=True)
# bbc_tweets.drop(bbc_tweets.columns[[22, 24]], axis=1, inplace=True)
#
# # Convert the 'date' column to datetime
# cnn_tweets['date'] = pd.to_datetime(cnn_tweets['date'], errors='coerce')
# bbc_tweets['date'] = pd.to_datetime(bbc_tweets['date'], errors='coerce')
#
# # Define the date range
# start_date = pd.to_datetime('2016-01-01')
# end_date = pd.to_datetime('2020-01-01')
#
# # Filter by date range
# cnn_filtered = cnn_tweets[(cnn_tweets['date'] >= start_date) & (cnn_tweets['date'] <= end_date)]
# bbc_filtered = bbc_tweets[(bbc_tweets['date'] >= start_date) & (bbc_tweets['date'] <= end_date)]
#
# # Find the intersection of dates within this range
# cnn_dates = set(cnn_filtered['date'].dt.date)
# bbc_dates = set(bbc_filtered['date'].dt.date)
# common_dates = cnn_dates.intersection(bbc_dates)
#
# # Filter by common dates
# cnn_filtered = cnn_filtered[cnn_filtered['date'].dt.date.isin(common_dates)]
# bbc_filtered = bbc_filtered[bbc_filtered['date'].dt.date.isin(common_dates)]
#
# # Sample tweets (sample as many as possible if there are fewer than 2500)
# cnn_sample = cnn_filtered.sample(n=min(2500, len(cnn_filtered)), random_state=42)
# bbc_sample = bbc_filtered.sample(n=min(2500, len(bbc_filtered)), random_state=42)
#
# # Print the number of tweets in each sample
# print(f"Number of tweets in CNN sample: {len(cnn_sample)}")
# print(f"Number of tweets in BBC sample: {len(bbc_sample)}\n")
# #
# #####  EDA BEFORE DROPPING COLUMNS  #####
#
# # 1. Number of Tweets Per Day
# cnn_sample['date_only'] = cnn_sample['date'].dt.date
# bbc_sample['date_only'] = bbc_sample['date'].dt.date
#
# # Calculate the total number of tweets per day for CNN and BBC
# cnn_tweets_per_day = cnn_sample['date_only'].value_counts().sort_index()
# bbc_tweets_per_day = bbc_sample['date_only'].value_counts().sort_index()
#
# # Print the average number of tweets per day
# print(f"Average number of CNN tweets per day: {cnn_tweets_per_day.mean():.2f}")
# print(f"Average number of BBC tweets per day: {bbc_tweets_per_day.mean():.2f}\n")
#
#
# plt.figure(figsize=(14, 6))
# sns.countplot(data=cnn_sample, x='date_only', color='blue', label='CNN', alpha=0.6)
# sns.countplot(data=bbc_sample, x='date_only', color='red', label='BBC', alpha=0.6)
# plt.xticks(rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Number of Tweets')
# plt.title('Number of Tweets Per Day')
# plt.legend()
# plt.show()
#
# # 2. Tweet Length Distribution
# cnn_sample['tweet_length'] = cnn_sample['tweet'].apply(len)
# bbc_sample['tweet_length'] = bbc_sample['tweet'].apply(len)
#
#
# # Calculate the average tweet length for CNN and BBC
# cnn_avg_tweet_length = cnn_sample['tweet_length'].mean()
# bbc_avg_tweet_length = bbc_sample['tweet_length'].mean()
#
# # Print the average tweet length
# print(f"Average tweet length for CNN: {cnn_avg_tweet_length:.2f} characters")
# print(f"Average tweet length for BBC: {bbc_avg_tweet_length:.2f} characters\n")
#
#
#
# plt.figure(figsize=(14, 6))
# sns.histplot(cnn_sample['tweet_length'], bins=30, color='blue', label='CNN', alpha=0.6)
# sns.histplot(bbc_sample['tweet_length'], bins=30, color='red', label='BBC', alpha=0.6)
# plt.xlabel('Tweet Length')
# plt.ylabel('Frequency')
# plt.title('Tweet Length Distribution')
# plt.legend()
# plt.show()
#
#
#
#
#
# #####  SENTIMENT ANALYSIS  #####
# # Initialize the VADER sentiment analyzer
# sia = SentimentIntensityAnalyzer()
#
# def analyze_sentiment(text):
#     return sia.polarity_scores(text)
#
# # Perform sentiment analysis on CNN tweets
# cnn_sample['sentiment_scores'] = cnn_sample['tweet'].apply(analyze_sentiment)
# cnn_sample['sentiment'] = cnn_sample['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))
#
# # Perform sentiment analysis on BBC tweets
# bbc_sample['sentiment_scores'] = bbc_sample['tweet'].apply(analyze_sentiment)
# bbc_sample['sentiment'] = bbc_sample['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))
#
# # Function to calculate sentiment distribution
# def sentiment_distribution(df):
#     return df['sentiments:'].value_counts(normalize=True) * 100
#
# # Calculate sentiment distributions
# cnn_sentiment_dist = sentiment_distribution(cnn_sample)
# bbc_sentiment_dist = sentiment_distribution(bbc_sample)
#
# # Print sentiment analysis results
# print("\nSentiment Analysis Results:")
# print("CNN Sentiment Distribution:")
# print(cnn_sentiment_dist)
# print("\nBBC Sentiment Distribution:")
# print(bbc_sentiment_dist)
#
# # Visualize sentiment distribution
# plt.figure(figsize=(10, 6))
# width = 0.35
# x = np.arange(3)
# plt.bar(x - width/2, cnn_sentiment_dist, width, label='CNN')
# plt.bar(x + width/2, bbc_sentiment_dist, width, label='BBC')
# plt.xlabel('Sentiment')
# plt.ylabel('Percentage')
# plt.title('Sentiment Distribution: CNN vs BBC')
# plt.xticks(x, ['Negative', 'Neutral', 'Positive'])
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Analyze sentiment trends over time
# cnn_sentiment_trend = cnn_sample.groupby(cnn_sample['date'].dt.to_period('M'))['sentiment_scores'].apply(lambda x: np.mean([i['compound'] for i in x]))
# bbc_sentiment_trend = bbc_sample.groupby(bbc_sample['date'].dt.to_period('M'))['sentiment_scores'].apply(lambda x: np.mean([i['compound'] for i in x]))
#
# # Visualize sentiment trends
# plt.figure(figsize=(12, 6))
# plt.plot(cnn_sentiment_trend.index.astype(str), cnn_sentiment_trend.values, label='CNN')
# plt.plot(bbc_sentiment_trend.index.astype(str), bbc_sentiment_trend.values, label='BBC')
# plt.xlabel('Date')
# plt.ylabel('Average Sentiment (Compound Score)')
# plt.title('Sentiment Trends Over Time: CNN vs BBC')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# #####  CONTINUE  #####
#
# #Drop unnecessary columns (only need the text of the tweets)
# cnn_sample = cnn_sample[['tweet']]
# bbc_sample = bbc_sample[['tweet']]
#
# # Save the sampled data to new CSV files
# cnn_sample.to_csv('C:\\Users\\yyyy\\Desktop\\NLP\\cnn_sample.csv', index=False)
# bbc_sample.to_csv('C:\\Users\\yyyy\\Desktop\\NLP\\bbc_sample.csv', index=False)

#####  DATA LOADING  #####

# Load your sampled data
cnn_sample = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\cnn_sample.csv')
bbc_sample = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\bbc_sample.csv')





#####  PREPROCESSING  #####

# Function for preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(str(text).lower())

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop words removal
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in lemmas if token not in stop_words and token.isalnum() and token != 'http']   ##drop HTTP BECAUSE OF LINKS IN TWEETS

    return ' '.join(cleaned_tokens)


# Apply preprocessing to the tweets
cnn_sample['processed_tweet'] = cnn_sample.iloc[:, 0].apply(preprocess_text)
bbc_sample['processed_tweet'] = bbc_sample.iloc[:, 0].apply(preprocess_text)


#
# #####  TF-IDF ANALYSIS  #####
#
# # TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
#
# cnn_tfidf = tfidf.fit_transform(cnn_sample['processed_tweet'])
# bbc_tfidf = tfidf.transform(bbc_sample['processed_tweet'])
#
# # Get feature names (words)
# feature_names = tfidf.get_feature_names_out()
#
# # Calculate the sum of TF-IDF values for each word
# cnn_word_importance = cnn_tfidf.sum(axis=0).A1
# bbc_word_importance = bbc_tfidf.sum(axis=0).A1
#
# # Create dictionaries of word importance
# cnn_word_dict = dict(zip(feature_names, cnn_word_importance))
# bbc_word_dict = dict(zip(feature_names, bbc_word_importance))
#
# # Sort words by importance
# cnn_sorted_words = sorted(cnn_word_dict.items(), key=lambda x: x[1], reverse=True)
# bbc_sorted_words = sorted(bbc_word_dict.items(), key=lambda x: x[1], reverse=True)
#
#
#
#
#
# #####  VISUALIZATION  #####
#
# # Function to plot bar chart
# def plot_top_words(sorted_words, title, n=20):
#     words, values = zip(*sorted_words[:n])
#     plt.figure(figsize=(12, 6))
#     plt.bar(words, values)
#     plt.title(title)
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
#
#
# # # Plot top words for CNN and BBC
# # plot_top_words(cnn_sorted_words, 'Top Words in CNN Tweets')
# # plot_top_words(bbc_sorted_words, 'Top Words in BBC Tweets')
#
#
# # Function to create word cloud
# def create_word_cloud(word_dict, title):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(title)
#     plt.show()
#
#
# # # Create word clouds for CNN and BBC
# # create_word_cloud(cnn_word_dict, 'Word Cloud for CNN Tweets')
# # create_word_cloud(bbc_word_dict, 'Word Cloud for BBC Tweets')
#
# # Print top 20 words for each source
# print("Top 20 words in CNN tweets:")
# print(cnn_sorted_words[:20])
# print("\nTop 20 words in BBC tweets:")
# print(bbc_sorted_words[:20])
#
#
#
#
# #####  WORD2VEC  #####
#
# # Combine all processed tweets
# all_tweets = cnn_sample['processed_tweet'].tolist() + bbc_sample['processed_tweet'].tolist()
#
# # Train Word2Vec model
# w2v_model = Word2Vec(sentences=all_tweets, vector_size=100, window=5, min_count=1, workers=4)
#
# # Function to find similar words
# def find_similar_words(word, topn=10):
#     try:
#         similar_words = w2v_model.wv.most_similar(word, topn=topn)
#         return similar_words
#     except KeyError:
#         return []
#
# # Find similar words for top TF-IDF words
# top_tfidf_words = [word for word, _ in cnn_sorted_words[:10] + bbc_sorted_words[:10]]
# word2vec_similar = {word: find_similar_words(word) for word in top_tfidf_words}
#
#
#
# #####  AUTOENCODER  #####
#
# # Prepare data for autoencoder
# vocab = list(set(word for tweet in all_tweets for word in tweet))
# word_to_index = {word: i for i, word in enumerate(vocab)}
# index_to_word = {i: word for word, i in word_to_index.items()}
#
# # Convert tweets to numeric representation
# numeric_tweets = [[word_to_index[word] for word in tweet] for tweet in all_tweets]
#
# # Pad sequences
# max_length = max(len(tweet) for tweet in numeric_tweets)
# padded_tweets = tf.keras.preprocessing.sequence.pad_sequences(numeric_tweets, maxlen=max_length, padding='post')
#
# # Build autoencoder model
# input_dim = max_length
# encoding_dim = 32
#
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(encoding_dim, activation='relu')(input_layer)
# decoded = Dense(input_dim, activation='sigmoid')(encoded)
#
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#
# # Train autoencoder
# autoencoder.fit(padded_tweets, padded_tweets, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
#
# # Get the encoder part of the model
# encoder = Model(input_layer, encoded)
#
# # Function to get important words from autoencoder
# def get_important_words_autoencoder(tweets, top_n=20):
#     encoded_tweets = encoder.predict(tweets)
#     importance = np.sum(np.abs(encoded_tweets), axis=0)
#     top_indices = importance.argsort()[-top_n:][::-1]
#     return [index_to_word[i] for i in top_indices]
#
# important_words_autoencoder = get_important_words_autoencoder(padded_tweets)
#
#
# #####  COMPARISON OF RESULTS  #####
#
# print("Top 20 words by TF-IDF:")
# print("CNN:", [word for word, _ in cnn_sorted_words[:20]])
# print("BBC:", [word for word, _ in bbc_sorted_words[:20]])
#
# print("\nTop 20 words by Word2Vec similarity:")
# for word in top_tfidf_words[:20]:
#     print(f"{word}: {[similar[0] for similar in word2vec_similar[word][:5]]}")
#
# print("\nTop 20 words by Autoencoder:")
# print(important_words_autoencoder)
#


# #####  NAMED ENTITY RECOGNITION  #####
#
# def extract_entities(text):
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]
#
# cnn_entities = cnn_sample.iloc[:, 0].apply(extract_entities)
# bbc_entities = bbc_sample.iloc[:, 0].apply(extract_entities)
#
# # Count entity occurrences
# def count_entities(entities_series):
#     entity_counts = {}
#     for entities in entities_series:
#         for entity, label in entities:
#             if (entity, label) in entity_counts:
#                 entity_counts[(entity, label)] += 1
#             else:
#                 entity_counts[(entity, label)] = 1
#     return sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#
# cnn_top_entities = count_entities(cnn_entities)[:20]
# bbc_top_entities = count_entities(bbc_entities)[:20]
#
#
# print("\nTop 20 entities (NER):")
# print("CNN:", cnn_top_entities)
# print("BBC:", bbc_top_entities)



#####  RNN  #####

# Combine all processed tweets for RNN training
all_text = cnn_sample['processed_tweet'].tolist() + bbc_sample['processed_tweet'].tolist()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
total_words = len(tokenizer.word_index) + 1

# Convert text into sequences
input_sequences = []
for tweet in all_text:
    token_list = tokenizer.texts_to_sequences([tweet])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split predictors and labels
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the RNN model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=5, verbose=1)

# Function to generate text
def generate_text(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate text for CNN and BBC
seed_text_cnn = "Breaking news"
generated_text_cnn = generate_text(seed_text_cnn)
print("Generated CNN text:", generated_text_cnn)

seed_text_bbc = "Latest update"
generated_text_bbc = generate_text(seed_text_bbc)
print("Generated BBC text:", generated_text_bbc)


#####  GPT text generation  #####

def generate_gpt2_text(model, tokenizer, prompt_text, max_length=100):
    inputs = tokenizer.encode(prompt_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load pre-trained GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example usage
prompt_text = "The latest news from CNN: "
gpt2_generated_text = generate_gpt2_text(gpt2_model, gpt2_tokenizer, prompt_text)
print(gpt2_generated_text)







#####   OTHER STEPS  #####









# #####  TWEET SUMMARIZATION  #####
#
# # Initialize the summarization pipeline
# summarizer = pipeline('summarization')
#
# # Function to summarize tweets
# def summarize_tweet(tweet):
#     try:
#         # Summarize the tweet
#         summary = summarizer(tweet, max_length=15, min_length=10, do_sample=False)
#         return summary[0]['summary_text']
#     except Exception as e:
#         # If summarization fails, return the original tweet
#         return tweet
#
#
#
# # Function to print original and summarized tweets
# # Initialize the summarization pipeline
# summarizer = pipeline('summarization')
#
# # Function to summarize tweets
# def summarize_tweet(tweet):
#     try:
#         # Summarize the tweet
#         summary = summarizer(tweet, max_length=15, min_length=10, do_sample=False)
#         return summary[0]['summary_text']
#     except Exception as e:
#         # If summarization fails, return the original tweet
#         return tweet
#
#
# # Function to print original and summarized tweets
# def print_original_and_summary(df, num_samples=5):
#     samples = df.sample(n=num_samples, random_state=42)
#     for _, row in samples.iterrows():
#         print("Original tweet:")
#         print(row['tweet'])
#         print("\nSummarized tweet:")
#         print(row['summarized_tweet'])
#         print("\n" + "=" * 50 + "\n")
#
# # Print examples for CNN
# print("CNN Tweet Summarization Examples:")
# print_original_and_summary(cnn_sample)
#
# # Print examples for BBC
# print("\nBBC Tweet Summarization Examples:")
# print_original_and_summary(bbc_sample)
#
# #
# # # Apply summarization to both datasets
# # cnn_sample['summarized_tweet'] = cnn_sample['tweet'].apply(summarize_tweet)
# # bbc_sample['summarized_tweet'] = bbc_sample['tweet'].apply(summarize_tweet)
# #
# # # Calculate and print average reduction in length
# # cnn_reduction = (cnn_sample['tweet'].str.len() - cnn_sample['summarized_tweet'].str.len()) / cnn_sample['tweet'].str.len() * 100
# # bbc_reduction = (bbc_sample['tweet'].str.len() - bbc_sample['summarized_tweet'].str.len()) / bbc_sample['tweet'].str.len() * 100
# #
# # print(f"\nAverage length reduction for CNN tweets: {cnn_reduction.mean():.2f}%")
# # print(f"Average length reduction for BBC tweets: {bbc_reduction.mean():.2f}%")
# #
#
#





























###### CODE ABOVE







# WORKS_DROPED FOR _EDA
# #####  DATA EXTRACTION  #####
#
# # Load the datasets with low_memory=False to avoid dtype warnings
# cnn_tweets = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\tweets_cnn.csv', low_memory=False)
# bbc_tweets = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\tweets_bbc.csv', low_memory=False)
#
# # Drop problematic columns (22 and 24)
# cnn_tweets.drop(cnn_tweets.columns[[22, 24]], axis=1, inplace=True)
# bbc_tweets.drop(bbc_tweets.columns[[22, 24]], axis=1, inplace=True)
#
# # Convert the 'date' column to datetime
# cnn_tweets['date'] = pd.to_datetime(cnn_tweets['date'], errors='coerce')
# bbc_tweets['date'] = pd.to_datetime(bbc_tweets['date'], errors='coerce')
#
# # Define the date range
# start_date = pd.to_datetime('2016-01-01')
# end_date = pd.to_datetime('2020-01-01')
#
# # Filter by date range
# cnn_filtered = cnn_tweets[(cnn_tweets['date'] >= start_date) & (cnn_tweets['date'] <= end_date)]
# bbc_filtered = bbc_tweets[(bbc_tweets['date'] >= start_date) & (bbc_tweets['date'] <= end_date)]
#
# # Find the intersection of dates within this range
# cnn_dates = set(cnn_filtered['date'].dt.date)
# bbc_dates = set(bbc_filtered['date'].dt.date)
# common_dates = cnn_dates.intersection(bbc_dates)
#
# # Filter by common dates
# cnn_filtered = cnn_filtered[cnn_filtered['date'].dt.date.isin(common_dates)]
# bbc_filtered = bbc_filtered[bbc_filtered['date'].dt.date.isin(common_dates)]
#
# # Sample tweets (sample as many as possible if there are fewer than 2500)
# cnn_sample = cnn_filtered.sample(n=min(2500, len(cnn_filtered)), random_state=42)
# bbc_sample = bbc_filtered.sample(n=min(2500, len(bbc_filtered)), random_state=42)
#
# # Print the number of tweets in each sample
# print(f"Number of tweets in CNN sample: {len(cnn_sample)}")
# print(f"Number of tweets in BBC sample: {len(bbc_sample)}")
#
#
# # Drop unnecessary columns (only need the text of the tweets)
# cnn_sample = cnn_sample[['tweet']]
# bbc_sample = bbc_sample[['tweet']]
#
# # Save the sampled data to new CSV files
# cnn_sample.to_csv('C:\\Users\\yyyy\\Desktop\\NLP\\cnn_sample.csv', index=False)
# bbc_sample.to_csv('C:\\Users\\yyyy\\Desktop\\NLP\\bbc_sample.csv', index=False)
#
#
# #####  DATA LOADING  #####
#
# # Load your sampled data
# cnn_sample = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\cnn_sample.csv')
# bbc_sample = pd.read_csv('C:\\Users\\yyyy\\Desktop\\NLP\\bbc_sample.csv')
#















