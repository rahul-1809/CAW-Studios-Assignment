import os
import gdown
import sys

def download_large_files():
    """Download large CSV files from Google Drive that are too big for GitHub."""
    
    # Google Drive file IDs and their corresponding local filenames
    files_to_download = {
        '1G3LrNhQ_Aou5_H9AGuGFL5YD5SJAsoG-': 'rating.csv',
        '19RYqakQfY0dazo8uuNHxHFRgIVme7kKE': 'ratings_cleaned.csv'
    }
    
    print("📥 Downloading large CSV files from Google Drive...")
    print("=" * 50)
    
    for file_id, filename in files_to_download.items():
        if os.path.exists(filename):
            print(f"✅ {filename} already exists, skipping...")
            continue
            
        print(f"📥 Downloading {filename}...")
        try:
            # Construct the Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download the file
            gdown.download(url, filename, quiet=False)
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
                print(f"✅ Successfully downloaded {filename} ({file_size:.1f} MB)")
            else:
                print(f"❌ Failed to download {filename}")
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    print("\n✅ All large files downloaded successfully!")
    return True

def check_required_files():
    """Check if all required files are present."""
    # Only check for the large files that are downloaded from Google Drive
    required_files = [
        'rating.csv',
        'ratings_cleaned.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files are present!")
        return True

if __name__ == "__main__":
    print("🎬 Movie & Book Recommendation System - File Downloader")
    print("=" * 60)
    
    # First check what files are missing
    print("🔍 Checking for required files...")
    if check_required_files():
        print("\n🎉 All files are ready! You can now run the application.")
        sys.exit(0)
    
    # Download missing large files
    print("\n📥 Downloading missing large files...")
    if download_large_files():
        print("\n🔍 Final check...")
        if check_required_files():
            print("\n🎉 All files are ready! You can now run the application.")
            print("\nNext steps:")
            print("1. Run 'python data_preparation.py' to prepare movie data")
            print("2. Run 'python book_data_preparation.py' to prepare book data")
            print("3. Run 'python app.py' to start the recommendation system")
        else:
            print("\n❌ Some files are still missing. Please check the error messages above.")
    else:
        print("\n❌ Failed to download required files. Please check your internet connection and try again.") 