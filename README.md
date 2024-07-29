# Movie Recommendation System

This project implements an advanced hybrid movie recommendation system that combines content-based filtering, collaborative filtering, and natural language processing techniques to provide personalized movie recommendations.

## Features

- Content-based filtering using movie metadata (genres, cast, crew, keywords)
- Collaborative filtering using Singular Value Decomposition (SVD)
- Natural Language Processing (NLP) with GPT-3.5 for enhanced movie analysis
- Hybrid recommendation system combining multiple techniques
- Performance evaluation using Root Mean Square Error (RMSE)

## Data Sources

The project uses the following datasets:
- Movies metadata
- User ratings
- Movie credits
- Movie keywords

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- nltk
- openai

## Setup and Installation

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Set up your OpenAI API key (see below)
4. Download the required datasets and update the file paths in the code

### Setting up OpenAI API Key

To use the GPT-3.5 model for enhanced movie analysis, you need to obtain an API key from OpenAI:

1. Sign up for an account at [OpenAI](https://openai.com/)
2. Navigate to the API section and create a new API key
3. In the code, replace the placeholder API key with your actual key:

```python
api_call = OpenAI(api_key="your_api_key_here")
```

**Note:** Keep your API key confidential and never share it publicly.

## Usage

Run the main script to see example recommendations:

```python
python movie_recommender.py
```

## How the Recommender Works

The movie recommendation system uses a hybrid approach, combining several techniques:

1. **Content-Based Filtering:**
   - Creates a "soup" of movie features including genres, cast, crew, and keywords
   - Uses TF-IDF vectorization to convert text data into numerical form
   - Calculates cosine similarity between movies based on these features

2. **GPT-Enhanced Analysis:**
   - Utilizes OpenAI's GPT-3.5 model to generate a concise analysis of each movie
   - This analysis is added to the feature "soup" to enrich the content-based filtering

3. **Collaborative Filtering:**
   - Applies Singular Value Decomposition (SVD) to the user-item rating matrix
   - Captures latent factors that represent user preferences and movie characteristics
   - Uses these factors to predict user ratings for unseen movies

4. **Hybrid Recommendations:**
   - Combines the results from content-based filtering and collaborative filtering
   - Weights the importance of each method (e.g., 30% content-based, 70% collaborative)
   - Ranks movies based on this combined score

5. **Performance Evaluation:**
   - Calculates Root Mean Square Error (RMSE) to assess the accuracy of rating predictions

The system first finds similar movies using content-based filtering, then adjusts these recommendations based on predicted user ratings from collaborative filtering. The GPT-enhanced analysis provides additional context and depth to the recommendations.

## Results

The hybrid system provides personalized movie recommendations that take into account both movie content and user preferences. The GPT-enhanced analysis adds an extra layer of insight into each recommended movie.

Example output:
```
Model RMSE: 0.8976
Recommendations for 'The Dark Knight':
1. Batman Begins (2005) - Estimated Rating: 4.2
2. The Dark Knight Rises (2012) - Estimated Rating: 4.1
3. Inception (2010) - Estimated Rating: 4.0
...
```

## Future Improvements

- Incorporate more advanced NLP techniques for text analysis
- Experiment with deep learning models for collaborative filtering
- Implement a user interface for interactive recommendations
