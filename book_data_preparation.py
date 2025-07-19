import pandas as pd
import numpy as np

def load_and_prepare_book_data():
    """Load and prepare Book-Crossing data from the root directory."""
    
    # File paths
    books_file = 'BX-Books.csv'
    ratings_file = 'BX-Book-Ratings.csv'
    users_file = 'BX-Users.csv'
    
    print(f"📁 Loading books from: {books_file}")
    print(f"📁 Loading ratings from: {ratings_file}")
    print(f"📁 Loading users from: {users_file}")
    
    try:
        # Load the data with appropriate separators
        # Book-Crossing data often uses semicolon as separator
        try:
            books_df = pd.read_csv(books_file, sep=';', encoding='latin-1')
        except:
            books_df = pd.read_csv(books_file, encoding='latin-1')
        
        try:
            ratings_df = pd.read_csv(ratings_file, sep=';', encoding='latin-1')
        except:
            ratings_df = pd.read_csv(ratings_file, encoding='latin-1')
        
        try:
            users_df = pd.read_csv(users_file, sep=';', encoding='latin-1')
        except:
            users_df = pd.read_csv(users_file, encoding='latin-1')
        
        print(f"✅ Loaded {len(books_df)} books and {len(ratings_df)} ratings")
        print(f"✅ Loaded {len(users_df)} users")
        
        # Clean and standardize column names
        print("🧹 Cleaning and standardizing data...")
        
        # Standardize book column names
        book_columns_mapping = {
            'ISBN': 'ISBN',
            'Book-Title': 'Book-Title',
            'Book-Author': 'Book-Author',
            'Year-Of-Publication': 'Year-Of-Publication',
            'Publisher': 'Publisher',
            'Image-URL-S': 'Image-URL-S',
            'Image-URL-M': 'Image-URL-M',
            'Image-URL-L': 'Image-URL-L'
        }
        
        # Rename columns if they exist
        for old_col, new_col in book_columns_mapping.items():
            if old_col in books_df.columns:
                books_df = books_df.rename(columns={old_col: new_col})
        
        # Standardize rating column names
        rating_columns_mapping = {
            'User-ID': 'User-ID',
            'ISBN': 'ISBN',
            'Book-Rating': 'Book-Rating'
        }
        
        for old_col, new_col in rating_columns_mapping.items():
            if old_col in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={old_col: new_col})
        
        # Clean data
        print("🧹 Cleaning data...")
        
        # Remove rows with missing values
        books_df = books_df.dropna(subset=['ISBN', 'Book-Title'])
        ratings_df = ratings_df.dropna()
        
        # Convert ratings to numeric, removing non-numeric values
        ratings_df['Book-Rating'] = pd.to_numeric(ratings_df['Book-Rating'], errors='coerce')
        ratings_df = ratings_df.dropna(subset=['Book-Rating'])
        
        # Filter out ratings of 0 (often indicates unrated books)
        ratings_df = ratings_df[ratings_df['Book-Rating'] > 0]
        
        # Convert User-ID to numeric
        ratings_df['User-ID'] = pd.to_numeric(ratings_df['User-ID'], errors='coerce')
        ratings_df = ratings_df.dropna(subset=['User-ID'])
        
        # Convert Year-Of-Publication to numeric
        books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
        
        # Filter out invalid years
        books_df = books_df[
            (books_df['Year-Of-Publication'] >= 1900) & 
            (books_df['Year-Of-Publication'] <= 2024)
        ]
        
        # Calculate book statistics
        print("📊 Calculating book statistics...")
        book_stats = ratings_df.groupby('ISBN').agg({
            'Book-Rating': ['mean', 'count']
        }).reset_index()
        book_stats.columns = ['ISBN', 'avg_rating', 'num_ratings']
        
        # Save book_stats to root directory
        book_stats_path = "book_stats.csv"
        book_stats.to_csv(book_stats_path, index=False)
        print(f"✅ Book stats saved to: {book_stats_path}")
        
        # Merge with books dataframe
        books_df = books_df.merge(book_stats, on='ISBN', how='left')
        
        # Fill NaN values
        books_df['avg_rating'] = books_df['avg_rating'].fillna(0)
        books_df['num_ratings'] = books_df['num_ratings'].fillna(0)
        
        # Filter books with at least some ratings for better recommendations
        books_df = books_df[books_df['num_ratings'] >= 1]
        
        # Create a mapping of books to their genres based on title keywords
        print("📚 Creating genre mapping...")
        genre_keywords = {
            'Fiction': ['fiction', 'novel', 'story', 'tale'],
            'Mystery': ['mystery', 'detective', 'crime', 'thriller'],
            'Romance': ['romance', 'love', 'romantic'],
            'Science Fiction': ['science fiction', 'sci-fi', 'space', 'future'],
            'Fantasy': ['fantasy', 'magic', 'wizard', 'dragon'],
            'Biography': ['biography', 'autobiography', 'memoir'],
            'History': ['history', 'historical', 'war', 'battle'],
            'Self-Help': ['self-help', 'motivation', 'success', 'personal'],
            'Business': ['business', 'management', 'economics', 'finance'],
            'Technology': ['technology', 'computer', 'programming', 'software']
        }
        
        # Add genre columns
        for genre, keywords in genre_keywords.items():
            books_df[f'genre_{genre}'] = books_df['Book-Title'].str.lower().apply(
                lambda x: any(keyword in str(x).lower() for keyword in keywords)
            ).astype(int)
        
        # Save processed data to root directory
        processed_books_path = "books_cleaned.csv"
        processed_ratings_path = "book_ratings_cleaned.csv"
        
        books_df.to_csv(processed_books_path, index=False)
        ratings_df.to_csv(processed_ratings_path, index=False)
        
        print(f"✅ Processed data saved to:")
        print(f"   - {processed_books_path}")
        print(f"   - {processed_ratings_path}")
        
        return books_df, ratings_df, users_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

if __name__ == "__main__":
    print("📚 Book-Crossing Data Preparation")
    print("=" * 40)
    
    books_df, ratings_df, users_df = load_and_prepare_book_data()
    
    if books_df is not None and ratings_df is not None:
        print(f"\n📊 Data Summary:")
        print(f"   - Books: {len(books_df)}")
        print(f"   - Ratings: {len(ratings_df)}")
        print(f"   - Users: {ratings_df['User-ID'].nunique()}")
        print(f"   - Genres: {len([col for col in books_df.columns if col.startswith('genre_')])}")
        print(f"   - Average rating: {ratings_df['Book-Rating'].mean():.2f}")
        print(f"   - Rating range: {ratings_df['Book-Rating'].min()} - {ratings_df['Book-Rating'].max()}")
        print(f"   - Books with cover images: {books_df['Image-URL-M'].notna().sum()}")
        
        print("\n✅ Book data preparation completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the recommendation system")
    else:
        print("\n❌ Book data preparation failed. Please check the error messages above.") 