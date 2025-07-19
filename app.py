from flask import Flask, render_template, request, redirect, url_for, session, flash
from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from gemini_client import gemini_generate_content
from book_collaborative_filtering import (
    load_book_data, filter_active, build_sparse_user_book_matrix,
    get_book_recommendations_sparse, get_top_books, find_matching_book
)

app = Flask(__name__)
app.secret_key = '32'  

# Load movie data once at startup with memory-efficient dtypes
movies_df = pd.read_csv(
    'movies_cleaned.csv',
    dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'}
)

# Check if ratings_cleaned.csv exists, if not, try to download it
if not os.path.exists('ratings_cleaned.csv'):
    print("⚠️  ratings_cleaned.csv not found. Attempting to download from Google Drive...")
    try:
        import gdown
        url = "https://drive.google.com/uc?id=19RYqakQfY0dazo8uuNHxHFRgIVme7kKE"
        gdown.download(url, 'ratings_cleaned.csv', quiet=False)
        if not os.path.exists('ratings_cleaned.csv'):
            raise Exception("Failed to download ratings_cleaned.csv")
        print("✅ Successfully downloaded ratings_cleaned.csv")
    except Exception as e:
        print(f"❌ Error downloading ratings_cleaned.csv: {e}")
        print("Please run 'python download_large_files.py' to download required files.")
        sys.exit(1)

ratings_df = pd.read_csv(
    'ratings_cleaned.csv',
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32', 'timestamp': 'str'},
    low_memory=True
)

# Load book data and collaborative filtering model
books_df, book_ratings_df, book_stats = load_book_data()
filtered_book_ratings = filter_active(book_ratings_df, min_user_ratings=10, min_book_ratings=10)
book_matrix, user_id_to_idx, book_isbn_to_idx, book_user_ids, book_isbns = build_sparse_user_book_matrix(filtered_book_ratings)

# Helper functions for movies (unchanged)
def get_genre_columns(movies_df):
    base_cols = {'movieId', 'title', 'genres'}
    genre_columns = [col for col in movies_df.columns if col not in base_cols]
    return genre_columns

def build_user_profile(ratings_df, movies_df, user_id, genre_columns, min_rating=4.0):
    user_rated = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= min_rating)]
    user_movies = pd.merge(user_rated, movies_df, on='movieId')
    if user_movies.empty:
        return np.zeros(len(genre_columns))
    user_profile = user_movies[genre_columns].mean().values
    return user_profile

def recommend_movies(user_profile, movies_df, genre_columns, n=10, seen_movie_ids=None):
    if seen_movie_ids is None:
        seen_movie_ids = set()
    similarities = cosine_similarity([user_profile], movies_df[genre_columns].values)[0]
    movies_df = movies_df.copy()
    movies_df['similarity'] = similarities
    recs = movies_df[~movies_df['movieId'].isin(seen_movie_ids)].sort_values('similarity', ascending=False)
    return recs[['movieId', 'title', 'genres', 'similarity']].head(n)

def get_cold_start_recommendations(movies_df, ratings_df, min_ratings=1000, n=10):
    movie_stats = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'num_ratings']
    movie_stats = movie_stats.reset_index()
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    top_movies = pd.merge(popular_movies, movies_df, on='movieId')
    top_movies = top_movies.sort_values('avg_rating', ascending=False)
    return top_movies[['movieId', 'title', 'genres', 'avg_rating', 'num_ratings']].head(n)

def explain_movie_book_pair(user_profile, movie_row, book_row, genre_columns):
    # Only include genres that exist in the movie_row
    genres = ', '.join([g for g in genre_columns if g in movie_row and movie_row[g] == 1])
    book_title = book_row.get('Book-Title', '')
    book_author = book_row.get('Book-Author', '')
    prompt = (
        f"Explain in English why this movie and book are recommended together. "
        f"User's genre preferences (0-1 scale): {dict(zip(genre_columns, user_profile.round(2)))}. "
        f"Movie: {movie_row['title']} (Genres: {genres}). "
        f"Book: {book_title} by {book_author}. "
        f"Be concise and friendly."
    )
    return gemini_generate_content(prompt)

# Helper: get books already rated by user
def get_rated_books(user_id, book_ratings_df):
    return set(book_ratings_df[book_ratings_df['User-ID'] == user_id]['ISBN'])

# Recommend top N unique books by genre, excluding already rated and already recommended in this batch
def find_matching_books(movie_genres, books_df, book_stats, rated_books, already_recommended, min_ratings=50, n=3):
    matches = pd.DataFrame()
    for genre in movie_genres:
        genre_matches = books_df[books_df['Book-Title'].str.contains(genre, case=False, na=False)]
        matches = pd.concat([matches, genre_matches])
    if not matches.empty and 'num_ratings' in matches.columns:
        matches = pd.merge(matches, book_stats, on='ISBN', suffixes=('', '_stats'))
        if 'num_ratings' in matches.columns:
            matches = matches[matches['num_ratings'] >= min_ratings]
        matches = matches[~matches['ISBN'].isin(rated_books | already_recommended)]
        if not matches.empty:
            return matches.sort_values('avg_rating', ascending=False).head(n).to_dict('records')
    # Fallback: top-rated books not yet rated or recommended
    top_books = get_top_books(book_stats, books_df, min_ratings=min_ratings, n=n+len(rated_books|already_recommended))
    top_books = [b for b in top_books if b['ISBN'] not in rated_books and b['ISBN'] not in already_recommended][:n]
    return top_books

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        session['user_id'] = user_id
        return redirect(url_for('recommend'))
    return render_template('home.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    user_id = session.get('user_id', None)
    if user_id is None:
        return redirect(url_for('home'))
    global ratings_df, book_ratings_df
    genre_columns = get_genre_columns(movies_df)
    seen_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    is_new_user = user_ratings.empty
    cold_start_message = None
    rated_books = get_rated_books(user_id, book_ratings_df)
    movie_book_pairs = []
    n_books_per_movie = 3
    # Handle rating submission
    if request.method == 'POST':
        rate_type = request.form.get('rate_type')
        rating = request.form.get('rating')
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            flash('Please enter a valid rating between 1 and 5.')
            return redirect(url_for('recommend'))
        if not (1.0 <= rating <= 5.0):
            flash('Please enter a rating between 1 and 5.')
            return redirect(url_for('recommend'))
        if rate_type == 'movie':
            movie_id = int(request.form.get('movie_id'))
            new_row = pd.DataFrame({'userId': [user_id], 'movieId': [movie_id], 'rating': [rating], 'timestamp': [pd.Timestamp.now()]})
            ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
        elif rate_type == 'book':
            isbn = request.form.get('isbn')
            new_row = pd.DataFrame({'User-ID': [user_id], 'ISBN': [isbn], 'Book-Rating': [int(rating)]})
            book_ratings_df = pd.concat([book_ratings_df, new_row], ignore_index=True)
        return redirect(url_for('recommend'))
    # Generate recommendations (after any rating update)
    already_recommended_books = set()
    if is_new_user:
        movie_recs = get_cold_start_recommendations(movies_df, ratings_df, min_ratings=1000, n=5)
        cold_start_message = "You are a new user! Here are some highly rated movies and books to get you started."
        top_books = get_top_books(book_stats, books_df, min_ratings=50, n=n_books_per_movie + len(rated_books))
        top_books = [b for b in top_books if b['ISBN'] not in rated_books][:n_books_per_movie]
        for _, movie_row in movie_recs.iterrows():
            explanation = (
                "**Why this pair?**\n\n" +
                "- The movie and books are highly rated by many users.\n" +
                "- These books are selected to give you a variety of top choices to start your reading journey!"
            )
            movie_book_pairs.append({
                'movie': movie_row,
                'books': top_books,
                'explanation': explanation
            })
            already_recommended_books.update([b['ISBN'] for b in top_books])
    else:
        user_profile = build_user_profile(ratings_df, movies_df, user_id, genre_columns)
        movie_recs = recommend_movies(user_profile, movies_df, genre_columns, n=5, seen_movie_ids=seen_movie_ids)
        for _, movie_row in movie_recs.iterrows():
            movie_genres = movie_row['genres'].split('|') if isinstance(movie_row['genres'], str) else []
            books = find_matching_books(movie_genres, books_df, book_stats, rated_books, already_recommended_books, min_ratings=50, n=n_books_per_movie)
            already_recommended_books.update([b['ISBN'] for b in books])
            book_titles = [b['Book-Title'] for b in books]
            prompt = (
                "You are a helpful recommender system. Provide a brief, friendly, and well-structured explanation in Markdown (2-3 bullet points max) for why this movie and these books are recommended together. "
                "Use bullet points for each reason.\n"
                f"User's genre preferences: {dict(zip(genre_columns, user_profile.round(2)))}. "
                f"Movie: {movie_row['title']} (Genres: {', '.join(movie_genres)}). "
                f"Books: {book_titles}. "
                "Focus on genre, themes, and what the user might enjoy."
            )
            explanation = gemini_generate_content(prompt)
            movie_book_pairs.append({
                'movie': movie_row,
                'books': books,
                'explanation': explanation
            })
    return render_template('recommend.html', movie_book_pairs=movie_book_pairs, cold_start_message=cold_start_message)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        max_user_id = ratings_df['userId'].max() if not ratings_df.empty else 0
        new_user_id = int(max_user_id) + 1
        session['user_id'] = new_user_id
        flash(f'New user created! Your User ID is {new_user_id}.')
        return redirect(url_for('recommend'))
    return render_template('add_user.html')

@app.route('/nl_query', methods=['GET', 'POST'])
def nl_query():
    suggestions = None
    if request.method == 'POST':
        movie_name = request.form.get('movie_name', '').strip()
        description = request.form.get('description', '').strip()
        prompt = (
            "You are a helpful recommender system. Provide a brief, friendly, and well-structured list in Markdown (2-3 bullet points max) for movies that match the user's request. "
            "Use bullet points for each suggestion.\n"
            f"Reference movie: '{movie_name}'. User's description: '{description}'. "
            f"Here are the available movies (title and genres):\n" +
            '\n'.join(f"- {row['title']} ({row['genres']})" for _, row in movies_df.iterrows()) +
            "\nFor each suggestion, provide a short explanation."
        )
        suggestions = gemini_generate_content(prompt, max_tokens=512)
    return render_template('nl_query.html', suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True) 