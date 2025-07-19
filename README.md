# Movie & Book Recommendation System

A comprehensive recommendation system that combines **movies** (MovieLens 20M dataset) and **books** (Book-Crossing dataset) with AI-powered explanations and natural language queries using Gemini API.

## 🎯 Features

### **Movie Recommendations**
- **Content-based filtering** using genre similarity and cosine similarity
- **Cold start recommendations** for new users (popular, highly-rated movies)
- **Personalized recommendations** based on user ratings and preferences
- **Dynamic updates** - recommendations refresh after each rating

### **Book Recommendations**
- **Collaborative filtering** using TruncatedSVD on user-book rating matrices
- **Sparse matrix optimization** for efficient handling of large datasets
- **Popularity-based fallback** when collaborative filtering data is insufficient

### **Cross-Domain Recommendations**
- **Movie-Book Pairs** - Each movie recommendation comes with matching books
- **Genre-based matching** - Books are selected based on movie genres
- **Unique recommendations** - No duplicate books across different movies
- **Visual book covers** - Display book cover images from the dataset

### **AI-Powered Features**
- **Gemini API integration** for intelligent explanations
- **Natural language queries** - Describe what you want to watch/read
- **Structured explanations** with bullet points and markdown formatting
- **Concise, readable responses** optimized for user experience

### **Interactive Web Interface**
- **Modern, responsive design** with card-based layout
- **Real-time rating system** for both movies and books (1-5 scale)
- **Dynamic recommendation updates** after each rating
- **User management** - create new users or use existing IDs
- **Navigation between features** - home, recommendations, and NL queries

## 🛠️ Tech Stack

### **Backend**
- **Python 3.12** with Flask web framework
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for collaborative filtering (TruncatedSVD)
- **Requests** for Gemini API integration

### **Frontend**
- **HTML5** with Jinja2 templating
- **CSS3** with modern styling (flexbox, gradients, shadows)
- **Responsive design** for mobile and desktop

### **Data Processing**
- **Sparse matrices** for efficient collaborative filtering
- **Cosine similarity** for content-based movie recommendations
- **Data cleaning and preprocessing** scripts included

### **AI Integration**
- **Google Gemini API** for natural language processing
- **Structured prompts** for consistent, readable explanations

## 📁 Project Structure

```
caw_studios/
├── app.py                          # Main Flask application
├── gemini_client.py                # Gemini API client
├── data_preparation.py             # MovieLens data preprocessing
├── cold_start_recommendation.py    # Cold start movie recommendations
├── book_data_preparation.py        # Book-Crossing data preprocessing
├── book_collaborative_filtering.py # Book recommendation algorithms
├── interactive_personalized_recommendation.py # Interactive CLI recommendations
├── templates/                      # Jinja2 HTML templates
│   ├── home.html                   # Main landing page
│   ├── add_user.html              # User creation/selection
│   ├── recommend.html             # Movie-book recommendations
│   └── nl_query.html              # Natural language query interface
├── movie.csv                       # MovieLens movies dataset
├── rating.csv                      # MovieLens ratings dataset (downloaded from Google Drive)
├── BX-Books.csv                    # Book-Crossing books dataset
├── BX-Book-Ratings.csv             # Book-Crossing ratings dataset
├── BX-Users.csv                    # Book-Crossing users dataset
├── movies_cleaned.csv              # Processed MovieLens movies
├── ratings_cleaned.csv             # Processed MovieLens ratings (downloaded from Google Drive)
├── books_cleaned.csv               # Processed Book-Crossing books
├── book_ratings_cleaned.csv        # Processed Book-Crossing ratings
├── book_stats.csv                  # Book-Crossing book statistics
├── download_large_files.py         # Script to download large files from Google Drive
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Setup Instructions

### **1. Environment Setup**
```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Data Preparation**
The large CSV files (`rating.csv` and `ratings_cleaned.csv`) are stored on Google Drive due to GitHub file size limits.

**Option A: Automatic Download (Recommended)**
```bash
python download_large_files.py
```

**Option B: Manual Setup**
Place the following files in the project root:
- `movie.csv` (MovieLens 20M)
- `BX-Books.csv`, `BX-Book-Ratings.csv`, `BX-Users.csv` (Book-Crossing)

Download large files from Google Drive:
- [rating.csv](https://drive.google.com/file/d/1G3LrNhQ_Aou5_H9AGuGFL5YD5SJAsoG-/view?usp=sharing)
- [ratings_cleaned.csv](https://drive.google.com/file/d/19RYqakQfY0dazo8uuNHxHFRgIVme7kKE/view?usp=sharing)

Then run the data preparation scripts:
```bash
python data_preparation.py
python book_data_preparation.py
```

### **3. Gemini API Setup**
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
   Or set the environment variable in your shell.

### **4. Launch the Application**
```bash
python app.py
```
Open your browser to: [http://localhost:5000](http://localhost:5000)

## 📖 Usage Guide

### **Getting Started**
1. **Home Page** - Enter your user ID or create a new user
2. **View Recommendations** - See personalized movie-book pairs with explanations
3. **Rate Content** - Rate movies and books to improve recommendations
4. **Natural Language Queries** - Ask for specific types of content

### **Recommendation Types**
- **Cold Start**: Popular movies for new users
- **Personalized**: Based on your rating history
- **Cross-Domain**: Movies paired with genre-matching books
- **Collaborative**: Books recommended by similar users

### **Rating System**
- **Scale**: 1-5 (including decimals like 3.5)
- **Real-time Updates**: Recommendations refresh after each rating
- **Both Domains**: Rate movies and books independently

## 🔧 Technical Details

### **Recommendation Algorithms**
1. **Movie Recommendations**:
   - Content-based filtering using genre one-hot encoding
   - Cosine similarity for finding similar movies
   - Popularity-based cold start for new users

2. **Book Recommendations**:
   - Collaborative filtering using TruncatedSVD
   - Sparse matrix operations for efficiency
   - User similarity-based recommendations

3. **Cross-Domain Matching**:
   - Genre-based book selection for movies
   - Popularity and rating-based prioritization
   - Duplicate prevention across recommendations

### **Performance Optimizations**
- **Sparse matrices** for large rating datasets
- **Efficient data structures** for fast similarity calculations
- **Memory management** for handling large datasets

## 🎨 UI Features

### **Visual Design**
- **Modern card-based layout** with shadows and rounded corners
- **Responsive design** that works on all screen sizes
- **Color-coded sections** for easy navigation
- **Book cover images** for visual appeal

### **User Experience**
- **Intuitive navigation** between different features
- **Real-time feedback** for all user actions
- **Clear explanations** with structured formatting
- **Easy rating system** with validation

## 🔒 Security & Production Notes

### **Current State (Development)**
- **Demo/Educational use** - no authentication required
- **Local storage** - data not persisted between sessions
- **API keys** stored in code (not secure for production)

### **Production Considerations**
- **Add user authentication** and session management
- **Implement secure API key storage** (environment variables)
- **Add database persistence** (MySQL/SQLite as preferred)
- **Implement rate limiting** for API calls
- **Add error handling** and logging
- **Deploy with proper WSGI server** (Gunicorn, uWSGI)

## 📊 Dataset Information

### **MovieLens 20M Dataset**
- **27,000+ movies** with genres and metadata
- **20+ million ratings** from 138,000+ users
- **Timestamps** for temporal analysis
- **Clean, well-structured** data format

### **Book-Crossing Dataset**
- **271,000+ books** with author and publisher info
- **1+ million ratings** from 278,000+ users
- **Book cover URLs** for visual display
- **ISBN-based** unique identification

## 🤝 Contributing

This is a demonstration project showcasing:
- **Recommendation system algorithms**
- **Cross-domain recommendation techniques**
- **AI integration for explanations**
- **Modern web application development**

Feel free to extend with additional features like:
- **More recommendation algorithms**
- **Additional datasets**
- **Advanced UI components**
- **Performance optimizations**

---

**Enjoy exploring movies and books with AI-powered recommendations!** 🎬📚 