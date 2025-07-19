import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gemini_client import gemini_generate_content

# Load cleaned data
def load_cleaned_data(movies_path='movies_cleaned.csv', ratings_path='ratings_cleaned.csv'):
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    return movies_df, ratings_df

# Get genre columns (one-hot columns)
def get_genre_columns(movies_df):
    base_cols = {'movieId', 'title', 'genres'}
    genre_columns = [col for col in movies_df.columns if col not in base_cols]
    return genre_columns

# Build user profile vector
def build_user_profile(ratings_df, movies_df, user_id, genre_columns, min_rating=4.0):
    user_rated = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= min_rating)]
    user_movies = pd.merge(user_rated, movies_df, on='movieId')
    if user_movies.empty:
        return np.zeros(len(genre_columns))
    user_profile = user_movies[genre_columns].mean().values
    return user_profile

# Recommend movies for user
def recommend_movies(user_profile, movies_df, genre_columns, n=10, seen_movie_ids=None):
    if seen_movie_ids is None:
        seen_movie_ids = set()
    similarities = cosine_similarity([user_profile], movies_df[genre_columns].values)[0]
    movies_df = movies_df.copy()
    movies_df['similarity'] = similarities
    recs = movies_df[~movies_df['movieId'].isin(seen_movie_ids)].sort_values('similarity', ascending=False)
    return recs[['movieId', 'title', 'genres', 'similarity']].head(n)

def explain_recommendation(user_profile, movie_row, genre_columns):
    # Compose a prompt for Gemini
    genres = ', '.join([g for g in genre_columns if movie_row[g] == 1])
    prompt = (
        f"Explain in English why the following movie is recommended to a user. "
        f"User's genre preferences (0-1 scale): {dict(zip(genre_columns, user_profile.round(2)))}. "
        f"Movie: {movie_row['title']} (Genres: {genres}). "
        f"Be concise and friendly."
    )
    return gemini_generate_content(prompt)

def natural_language_query(movies_df):
    movie_name = input("\nEnter a reference movie name (or leave blank to skip): ").strip()
    description = input("Describe the type of movie you want to watch: ").strip()
    if not description:
        print("No description provided. Returning to main menu.")
        return
    # Compose prompt for Gemini
    prompt = (
        f"Suggest 5 movies from the following list that match this request. "
        f"Reference movie: '{movie_name}'. User's description: '{description}'. "
        f"Here are the available movies (title and genres):\n" +
        '\n'.join(f"- {row['title']} ({row['genres']})" for _, row in movies_df.iterrows()) +
        "\nFor each suggestion, provide a short explanation in English."
    )
    response = gemini_generate_content(prompt, max_tokens=512)
    print("\nGemini suggestions:")
    print(response)

def interactive_recommendation_loop(user_id=1, n=10):
    movies_df, ratings_df = load_cleaned_data()
    genre_columns = get_genre_columns(movies_df)
    iteration = 1
    while True:
        print("\nOptions:")
        print("1. Get personalized recommendations")
        print("2. Natural language movie query")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == '0':
            print("Exiting recommendation loop.")
            break
        elif choice == '2':
            natural_language_query(movies_df)
            continue
        # Default: personalized recommendations
        seen_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
        user_profile = build_user_profile(ratings_df, movies_df, user_id, genre_columns)
        recommendations = recommend_movies(user_profile, movies_df, genre_columns, n=n, seen_movie_ids=seen_movie_ids)
        print(f"\nIteration {iteration}: Top {n} recommendations for user {user_id}:")
        for idx, row in recommendations.iterrows():
            explanation = explain_recommendation(user_profile, movies_df.loc[movies_df['movieId'] == row['movieId']].iloc[0], genre_columns)
            print(f"movieId: {row['movieId']}, Title: {row['title']}, Genres: {row['genres']}, Similarity: {row['similarity']:.3f}")
            print(f"  Why recommended: {explanation}")
        if recommendations.empty:
            print("No more recommendations available.")
            break
        # User selects a movie to watch
        try:
            selected_movie_id = int(input("\nEnter the movieId of the movie you want to watch (or 0 to stop): "))
        except ValueError:
            print("Invalid input. Stopping.")
            break
        if selected_movie_id == 0:
            print("Exiting recommendation loop.")
            break
        if selected_movie_id not in recommendations['movieId'].values:
            print("Selected movieId is not in the recommendations. Try again.")
            continue
        # User rates the movie
        try:
            rating = float(input(f"Rate the movie (1-5): "))
        except ValueError:
            print("Invalid rating. Skipping this round.")
            continue
        # Add new rating to ratings_df
        new_row = pd.DataFrame({'userId': [user_id], 'movieId': [selected_movie_id], 'rating': [rating], 'timestamp': [pd.Timestamp.now()]})
        ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
        iteration += 1
    print("\nFinal recommendations complete. Thank you!")

if __name__ == '__main__':
    user_id = 12  # Change this to test other users
    interactive_recommendation_loop(user_id=user_id, n=10) 