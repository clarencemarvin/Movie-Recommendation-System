import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from math import sqrt
import openai
from openai import OpenAI

import warnings; warnings.simplefilter('ignore')

# Set up OpenAI API
api_call = OpenAI(api_key="your_api_key_here")

# Load and preprocess data
md = pd.read_csv('/Users/clarencemarvin/Downloads/movies_metadata.csv')
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.95)

# Define weighted rating function
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

# Qualify movies
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

# Content-based filtering
links_small = pd.read_csv('/Users/clarencemarvin/Downloads/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]

# Merge additional data
credits = pd.read_csv('/Users/clarencemarvin/Downloads/credits.csv')
keywords = pd.read_csv('/Users/clarencemarvin/Downloads/keywords.csv')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]

# Process cast, crew, and keywords
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Clean and process text data
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x])

# Filter and stem keywords
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

stemmer = SnowballStemmer('english')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# Create soup of features
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

# Function to get movie analysis from GPT
def get_movie_analysis(title, year):
    prompt = f"Analyze the movie '{title}' ({year}). Describe its genre, themes, style, and why it's suitable for me in 2 concise sentences."
    
    completion = api_call.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a movie analysis expert."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip()

# Enhance the movie metadata with GPT analysis
def enhance_movie_data(smd):
    smd['gpt_analysis'] = smd.apply(lambda row: get_movie_analysis(row['title'], row['year']), axis=1)
    return smd

# Apply the enhancement (you might want to do this for a subset of movies first, as it can be time-consuming and costly)
smd = enhance_movie_data(smd)

# Create a new soup including the GPT analysis
smd['enhanced_soup'] = smd['soup'] + ' ' + smd['gpt_analysis']

# Create new count matrix and cosine similarity matrix
count = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
count_matrix = count.fit_transform(smd['enhanced_soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index and create indices
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# Define improved recommendation function
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'gpt_analysis']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

# Load ratings data
ratings = pd.read_csv('/Users/clarencemarvin/Downloads/ratings_small.csv')

# Prepare data for SVD
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Normalize the data
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

# Apply SVD
n_components = 300 
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item_matrix_scaled)
item_factors = svd.components_.T

# Function to get movie title from movieId
def get_movie_title(movieId):
    return smd[smd['id'] == movieId]['title'].iloc[0] if len(smd[smd['id'] == movieId]['title']) > 0 else None

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Function to predict rating
def predict_rating(user_id, movie_id):
    if user_id not in user_item_matrix.index or movie_id not in user_item_matrix.columns:
        return np.nan
    user_index = user_item_matrix.index.get_loc(user_id)
    movie_index = user_item_matrix.columns.get_loc(movie_id)
    predicted_rating = np.dot(user_factors[user_index], item_factors[movie_index])
    # Scale the predicted rating
    mean = scaler.mean_[movie_index]
    std = scaler.scale_[movie_index]
    return predicted_rating * std + mean

# Calculate RMSE
def calculate_model_rmse():
    actual_ratings = []
    predicted_ratings = []
    for _, row in ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        predicted_rating = predict_rating(user_id, movie_id)
        if not np.isnan(predicted_rating):
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)
    return calculate_rmse(actual_ratings, predicted_ratings)

# Hybrid recommendation function
def hybrid_recommendations(title, user_id, top_n=5):
    # Get content-based recommendations
    content_recs = improved_recommendations(title)
    
    # Collaborative filtering
    content_recs['est_rating'] = content_recs.index.map(lambda x: predict_rating(user_id, smd.loc[x, 'id']))
    
    # Combine scores
    content_recs['combined_score'] = (content_recs['wr'] * 0.3 + 
                                      content_recs['est_rating'] * 0.7)
    
    # Remove movies with NaN estimated ratings
    content_recs = content_recs[content_recs['est_rating'].notna()]
    
    # Sort and select top recommendations
    recommendations = content_recs.sort_values('combined_score', ascending=False).head(top_n)
    
    return recommendations[['title', 'year', 'vote_average', 'est_rating', 'combined_score', 'gpt_analysis']]

# Calculate and print RMSE
rmse = calculate_model_rmse()
print(f"Model RMSE: {rmse:.4f}")

# Example usage
print(hybrid_recommendations('The Dark Knight', 100))
