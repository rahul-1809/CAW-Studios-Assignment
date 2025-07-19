import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Load cleaned data
def load_book_data(books_path='books_cleaned.csv', ratings_path='book_ratings_cleaned.csv', stats_path='book_stats.csv'):
    books_df = pd.read_csv(books_path)
    ratings_df = pd.read_csv(ratings_path)
    book_stats = pd.read_csv(stats_path)
    return books_df, ratings_df, book_stats

# Filter to active users and popular books for memory efficiency
def filter_active(ratings_df, min_user_ratings=10, min_book_ratings=10):
    user_counts = ratings_df['User-ID'].value_counts()
    book_counts = ratings_df['ISBN'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_books = book_counts[book_counts >= min_book_ratings].index
    filtered = ratings_df[ratings_df['User-ID'].isin(active_users) & ratings_df['ISBN'].isin(popular_books)]
    return filtered

# Build sparse user-book matrix and index mappings
def build_sparse_user_book_matrix(ratings_df):
    user_ids = ratings_df['User-ID'].unique()
    book_isbns = ratings_df['ISBN'].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    book_isbn_to_idx = {isbn: idx for idx, isbn in enumerate(book_isbns)}
    row = ratings_df['User-ID'].map(user_id_to_idx)
    col = ratings_df['ISBN'].map(book_isbn_to_idx)
    data = ratings_df['Book-Rating'].astype(float)
    matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(book_isbns)))
    return matrix, user_id_to_idx, book_isbn_to_idx, user_ids, book_isbns

# Collaborative filtering using SVD on sparse matrix
def get_book_recommendations_sparse(user_id, matrix, user_id_to_idx, book_isbn_to_idx, user_ids, book_isbns, books_df, ratings_df, n=5):
    if user_id not in user_id_to_idx:
        return []
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(matrix)
    user_idx = user_id_to_idx[user_id]
    user_vector = latent_matrix[user_idx]
    book_vectors = svd.components_.T
    scores = np.dot(book_vectors, user_vector)
    # Exclude already rated books
    already_rated = set(ratings_df[ratings_df['User-ID'] == user_id]['ISBN'])
    recs = []
    for idx in np.argsort(scores)[::-1]:
        isbn = book_isbns[idx]
        if isbn not in already_rated:
            book_row = books_df[books_df['ISBN'] == isbn]
            if not book_row.empty:
                recs.append(book_row.iloc[0])
            if len(recs) >= n:
                break
    return recs

# For new users: recommend top-rated books
def get_top_books(book_stats, books_df, min_ratings=50, n=5):
    popular_books = book_stats[book_stats['num_ratings'] >= min_ratings]
    top_books = pd.merge(popular_books, books_df, on='ISBN')
    top_books = top_books.sort_values('avg_rating', ascending=False)
    return top_books.head(n).to_dict('records')

# Cross-domain: find a book matching a movie's genre (fallback to top-rated)
def find_matching_book(movie_genres, books_df, book_stats, min_ratings=50):
    for genre in movie_genres:
        matches = books_df[books_df['Book-Title'].str.contains(genre, case=False, na=False)]
        if not matches.empty:
            matches = pd.merge(matches, book_stats, on='ISBN')
            matches = matches[matches['num_ratings'] >= min_ratings]
            if not matches.empty:
                return matches.sort_values('avg_rating', ascending=False).iloc[0].to_dict()
    return get_top_books(book_stats, books_df, min_ratings=min_ratings, n=1)[0]

if __name__ == '__main__':
    books_df, ratings_df, book_stats = load_book_data()
    filtered_ratings = filter_active(ratings_df, min_user_ratings=10, min_book_ratings=10)
    matrix, user_id_to_idx, book_isbn_to_idx, user_ids, book_isbns = build_sparse_user_book_matrix(filtered_ratings)
    # Example usage:
    user_id = user_ids[0]  # Use a real/active user ID
    recs = get_book_recommendations_sparse(user_id, matrix, user_id_to_idx, book_isbn_to_idx, user_ids, book_isbns, books_df, filtered_ratings, n=5)
    print(f'Book recommendations for user {user_id}:')
    for book in recs:
        print(f"{book['Book-Title']} by {book['Book-Author']} (ISBN: {book['ISBN']})")
    # Example cross-domain:
    movie_genres = ['Science Fiction', 'Romance']
    match = find_matching_book(movie_genres, books_df, book_stats)
    print('\nMatching book for genres', movie_genres, ':')
    print(match) 