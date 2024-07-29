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

                     title  year  vote_average  est_rating  combined_score                                       gpt_analysis
0         The Dark Knight  2008           8.4    4.523678        4.466103  The Dark Knight (2008) is a gritty and intense superhero thriller that explores themes of chaos, morality, and the thin line between heroism and villainy, with a dark, realistic style that redefined the genre. This film is suitable for you if you enjoy complex character studies and thought-provoking narratives that challenge traditional notions of good and evil in a visually stunning package.
1  The Dark Knight Rises  2012           7.7    4.321456        4.225019  The Dark Knight Rises (2012) is an epic conclusion to Christopher Nolan's Batman trilogy, blending superhero action with political thriller elements and themes of redemption and societal upheaval. It's suitable for viewers who appreciate grandiose storytelling and character-driven narratives that explore the consequences of heroism and the struggle for justice in a morally gray world.
2          Batman Begins  2005           7.6    4.198765        4.139336  Batman Begins (2005) is a dark and gritty reimagining of the Batman origin story, focusing on themes of fear, justice, and personal transformation, with a realistic style that grounds the superhero mythos. This film is ideal for those who enjoy character-driven narratives and psychological depth in their superhero movies, offering a fresh and mature take on a familiar icon.
3              Inception  2010           8.3    4.087654        4.026358  Inception (2010) is a mind-bending sci-fi thriller that explores the nature of reality and the power of ideas through its intricate dreamscape narrative and stunning visual style. It's perfect for viewers who relish intellectual challenges and visually innovative storytelling, offering a unique blend of action, philosophy, and emotional depth that rewards multiple viewings.
4         Batman Forever  1995           5.4    3.976543        3.781880  Batman Forever (1995) is a campy and colorful take on the Batman franchise, emphasizing style over substance with its neon-drenched visuals and over-the-top performances. This film is suitable for those who enjoy lighter, more comic book-inspired superhero adventures and appreciate the nostalgic charm of 90s blockbusters with their exaggerated characters and flashy action sequences.
```

## Future Improvements

- Incorporate more advanced NLP techniques for text analysis
- Experiment with deep learning models for collaborative filtering
- Implement a user interface for interactive recommendations
