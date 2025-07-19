import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_and_prepare_data():
    """Load and prepare MovieLens data from the root directory."""
    
    # File paths
    movie_file = 'movie.csv'
    rating_file = 'rating.csv'
    
    print(f"📁 Loading movies from: {movie_file}")
    print(f"📁 Loading ratings from: {rating_file}")
    
    # Check if rating.csv exists, if not, try to download it
    if not os.path.exists(rating_file):
        print(f"⚠️  {rating_file} not found. Attempting to download from Google Drive...")
        try:
            import gdown
            url = "https://drive.google.com/uc?id=1G3LrNhQ_Aou5_H9AGuGFL5YD5SJAsoG-"
            gdown.download(url, rating_file, quiet=False)
            if not os.path.exists(rating_file):
                raise Exception("Failed to download rating.csv")
            print(f"✅ Successfully downloaded {rating_file}")
        except Exception as e:
            print(f"❌ Error downloading {rating_file}: {e}")
            print("Please run 'python download_large_files.py' to download required files.")
            return None, None
    
    try:
        # Load the data
        movies_df = pd.read_csv(movie_file)
        ratings_df = pd.read_csv(rating_file)
        
        print(f"✅ Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        # Basic data cleaning
        print("🧹 Cleaning data...")
        
        # Remove any rows with missing values
        movies_df = movies_df.dropna()
        ratings_df = ratings_df.dropna()
        
        # Ensure movieId is integer
        movies_df['movieId'] = movies_df['movieId'].astype(int)
        ratings_df['movieId'] = ratings_df['movieId'].astype(int)
        
        # Ensure userId is integer
        ratings_df['userId'] = ratings_df['userId'].astype(int)
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in ratings_df.columns:
            ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
        
        # Create genre one-hot encoding
        print("🎭 Creating genre encoding...")
        all_genres = set()
        for genres in movies_df['genres'].str.split('|'):
            if isinstance(genres, list):
                all_genres.update(genres)
        
        # Create one-hot encoding for genres
        for genre in sorted(all_genres):
            movies_df[f'genre_{genre}'] = movies_df['genres'].str.contains(genre, na=False).astype(int)
        
        # Calculate average ratings and number of ratings for each movie
        print("📊 Calculating movie statistics...")
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        # Merge with movies dataframe
        movies_df = movies_df.merge(movie_stats, on='movieId', how='left')
        
        # Fill NaN values
        movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
        movies_df['num_ratings'] = movies_df['num_ratings'].fillna(0)
        
        # Save processed data to root directory
        processed_movies_path = "movies_cleaned.csv"
        processed_ratings_path = "ratings_cleaned.csv"
        
        movies_df.to_csv(processed_movies_path, index=False)
        ratings_df.to_csv(processed_ratings_path, index=False)
        
        print(f"✅ Processed data saved to:")
        print(f"   - {processed_movies_path}")
        print(f"   - {processed_ratings_path}")
        
        return movies_df, ratings_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

if __name__ == "__main__":
    print("🎬 MovieLens Data Preparation")
    print("=" * 40)
    
    movies_df, ratings_df = load_and_prepare_data()
    
    if movies_df is not None and ratings_df is not None:
        print(f"\n📊 Data Summary:")
        print(f"   - Movies: {len(movies_df)}")
        print(f"   - Ratings: {len(ratings_df)}")
        print(f"   - Users: {ratings_df['userId'].nunique()}")
        print(f"   - Genres: {len([col for col in movies_df.columns if col.startswith('genre_')])}")
        print(f"   - Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"   - Rating range: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
        
        print("\n✅ Data preparation completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python book_data_preparation.py' to prepare book data")
        print("2. Run 'python app.py' to start the recommendation system")
    else:
        print("\n❌ Data preparation failed. Please check the error messages above.") 