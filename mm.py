import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# Load the MovieLens dataset
movies = pd.read_csv('movies.csv')  # Assuming 'genres' column exists in this file
ratings = pd.read_csv('ratings.csv')

# Split the dataset into training and testing data
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Create a user-item matrix for collaborative filtering
user_item_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Normalize the user-item matrix by subtracting the mean rating for each user
user_means = user_item_matrix.mean(axis=1)
user_item_matrix_normalized = user_item_matrix.sub(user_means, axis=0)

# Matrix Factorization using Truncated Singular Value Decomposition (SVD)
svd = TruncatedSVD(n_components=20, random_state=42)
U = svd.fit_transform(user_item_matrix_normalized)
sigma = np.diag(svd.singular_values_)
Vt = svd.components_

# Reconstruct the ratings matrix using the factorized matrices
predicted_ratings_normalized = np.dot(np.dot(U, sigma), Vt)

# Denormalize the predicted ratings by adding the user means
predicted_ratings = predicted_ratings_normalized + user_means.values[:, np.newaxis]

# Scale the predicted ratings to be within the range of the original ratings
scaler = MinMaxScaler(feature_range=(0.5, 5.0))
predicted_ratings_scaled = scaler.fit_transform(predicted_ratings)

# Convert the predicted ratings matrix back into a DataFrame
predicted_ratings_df = pd.DataFrame(predicted_ratings_scaled, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Function to get real-time movie details from TMDb
TMDB_API_KEY = 'bfb2ea1c2e1f358f711eb731c3878b5a'
TMDB_API_URL = 'https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US'

def get_movie_title(movie_id):
    """Fetches the movie title from TMDb using the movieId."""
    try:
        response = requests.get(TMDB_API_URL.format(movie_id, TMDB_API_KEY))
        if response.status_code == 200:
            movie_data = response.json()
            return movie_data['title']
        else:
            return None
    except:
        return None

# Function to filter movies by user's preferred genre
def get_movies_by_genre(genre):
    """Returns a list of movieIds that belong to the specified genre."""
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    return genre_movies['movieId'].tolist()

# Function to recommend movies for a specific user, filtered by genre
def get_movie_recommendations_by_genre(user_id, genre, num_recommendations=5):
    # Get all movie IDs that match the preferred genre
    genre_movie_ids = get_movies_by_genre(genre)
    
    # Get a list of movies the user has already rated
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    # Generate predictions for all movies in the selected genre that the user hasn't rated
    recommendations = []
    for movie_id in genre_movie_ids:
        if movie_id in predicted_ratings_df.columns and movie_id not in rated_movies:
            estimated_rating = predicted_ratings_df.loc[user_id, movie_id]
            recommendations.append((movie_id, estimated_rating))
    
    # Sort the predictions based on the estimated rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N movie recommendations
    top_recommendations = recommendations[:num_recommendations]
    
    # Fetch the movie titles for the recommendations using the TMDb API
    recommended_movies = []
    for movie_id, estimated_rating in top_recommendations:
        movie_title = get_movie_title(movie_id)
        if movie_title:
            recommended_movies.append((movie_title, estimated_rating))
    
    return recommended_movies

# Ask the user for their preferred genre
preferred_genre = input("Which type of movie do you like most? (e.g., Action, Comedy, Drama): ")

# Get movie recommendations for a user (e.g., user with ID 1) based on their preferred genre
user_id = 1
recommended_movies = get_movie_recommendations_by_genre(user_id, preferred_genre, num_recommendations=5)

# Print the recommended movies
print(f"Top 5 {preferred_genre} movie recommendations for User {user_id}:")
for movie, rating in recommended_movies:
    print(f"{movie} with an estimated rating of {rating:.2f}")
