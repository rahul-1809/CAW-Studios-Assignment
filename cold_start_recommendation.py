import pandas as pd

# Load cleaned data
def load_cleaned_data(movies_path='movies_cleaned.csv', ratings_path='ratings_cleaned.csv'):
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    return movies_df, ratings_df

# Compute top-N movies by average rating, with a minimum number of ratings
def get_top_movies(movies_df, ratings_df, min_ratings=1000, n=20):
    # Compute average rating and count for each movie
    movie_stats = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'num_ratings']
    movie_stats = movie_stats.reset_index()
    # Filter movies with enough ratings
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    # Join with movie metadata
    top_movies = pd.merge(popular_movies, movies_df, on='movieId')
    # Sort by average rating (descending)
    top_movies = top_movies.sort_values('avg_rating', ascending=False)
    return top_movies[['movieId', 'title', 'genres', 'avg_rating', 'num_ratings']].head(n)

if __name__ == '__main__':
    movies_df, ratings_df = load_cleaned_data()
    print('Loaded cleaned data.')
    top_movies = get_top_movies(movies_df, ratings_df, min_ratings=1000, n=20)
    print('\nTop 20 movies for new users (cold start):')
    print(top_movies) 