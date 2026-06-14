## Setup & Running
### 1. Prerequisites
Ensure you have Python 3.9+ installed. This project uses `uv` to manage python versions and virtual environments fast.
If `uv` is not installed, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 2. Install Dependencies
Create a virtual environment and install the required packages:
```bash
# Create virtual environment
uv venv --python 3.11
# Activate virtual environment
source .venv/bin/activate
# Install requirements
uv pip install -r requirements.txt
```
### 3. Fetch Dataset
Run the download script to retrieve and extract the official MovieLens small dataset. If offline, the script will automatically fallback to generating a clean, mock dataset.
```bash
python download_data.py
```
### 4. Run the App
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser and navigate to **`http://localhost:8501`** to use the application!
---
## Algorithms Explained
### 1. Content-Based Filtering
For each movie, we compile a text metadata profile:
$$\text{Metadata} = \text{Title} + \text{Genres} + \text{User-contributed tags}$$
We use `TfidfVectorizer` to convert this text into numeric TF-IDF vectors. The similarity between two movies $A$ and $B$ is computed using **Cosine Similarity**:
$$\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$
### 2. User-Based Collaborative Filtering
We build a User-Item rating matrix. For a new user, we compute their similarity with all existing users using Cosine Similarity on their ratings. We then predict the user's rating $\hat{R}_{u, i}$ for unrated movies using a similarity-weighted average of their $K$ nearest neighbors (similar users) ratings:
$$\hat{R}_{u, i} = \frac{\sum_{v \in \text{Neighbors}} \text{Sim}(u, v) \cdot R_{v, i}}{\sum_{v \in \text{Neighbors}} \text{Sim}(u, v)}$$
### 3. Hybrid Recommender
The hybrid model scores candidate movies by fusing Collaborative Filtering and Content-Based Filtering scores:
1. Candidate recommendations are extracted from Collaborative Filtering.
2. We compute the candidate's content similarity affinity score to the movies the active user rated $\ge 4.0$ stars.
3. We compute the final score using a weighted linear combination:
$$\text{Hybrid Score} = 0.6 \times \text{Normalized Predicted Rating} + 0.4 \times \text{Max Content Affinity}$$
This addresses the cold-start problem and leverages both user behaviors and item characteristics.

import streamlit as st
import pandas as pd
from recommender import MovieRecommender
# Set page config
st.set_page_config(
    page_title="CineMatch - Smart Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    /* Fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    /* Gradient Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #3b0764 50%, #090d16 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(167, 139, 250, 0.15);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        text-align: center;
    }
    .hero-title {
        background: linear-gradient(90deg, #c084fc 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }
    /* Glassmorphic Movie Cards */
    .movie-card {
        background: rgba(30, 41, 59, 0.35);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        margin-bottom: 1rem;
        min-height: 190px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.15);
        background: rgba(30, 41, 59, 0.5);
    }
    .movie-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    .genre-container {
        margin-bottom: 0.5rem;
    }
    .genre-pill {
        background: rgba(168, 85, 247, 0.15);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 9999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        display: inline-block;
        font-weight: 500;
    }
    .movie-metrics {
        font-size: 0.85rem;
        color: #94a3b8;
        display: flex;
        gap: 0.8rem;
        align-items: center;
        margin-top: auto;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    .score-tag {
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
    }
    
    .score-collab {
        background: rgba(168, 85, 247, 0.2);
        color: #e9d5ff;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    .score-content {
        background: rgba(34, 211, 238, 0.2);
        color: #cffafe;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }
    
    .score-hybrid {
        background: rgba(236, 72, 153, 0.2);
        color: #fbcfe8;
        border: 1px solid rgba(236, 72, 153, 0.3);
    }
    /* Custom sidebar elements */
    .sidebar-header {
        font-weight: 700;
        font-size: 1.3rem;
        color: #f8fafc;
        margin-bottom: 1rem;
    }
    
    .profile-empty {
        padding: 1rem;
        background: rgba(30, 41, 59, 0.25);
        border-radius: 8px;
        border: 1px dashed rgba(255, 255, 255, 0.1);
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .profile-item {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)
# Initialize Movie Recommender
@st.cache_resource
def get_recommender():
    return MovieRecommender()
try:
    recommender = get_recommender()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.info("Make sure you run the download script `python download_data.py` to prepare the data files.")
    st.stop()
# Initialize session state for user profile ratings
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
# --- SIDEBAR: Profile Builder ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">🎬 CineMatch Profile</div>', unsafe_allow_html=True)
    st.markdown("Rate movies to build your dynamic recommendation profile.")
    
    # 1. Search & Add Movie to Profile
    # Get movie titles sorted by popularity for a better search dropdown experience
    movies_sorted = recommender.movies_df.sort_values(by="rating_count", ascending=False)
    movie_options = dict(zip(movies_sorted["movieId"], movies_sorted["title"]))
    
    selected_mid = st.selectbox(
        "Search movie to rate:",
        options=list(movie_options.keys()),
        format_func=lambda x: movie_options[x],
        key="search_rate"
    )
    
    user_rating = st.slider("Your Rating:", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
    
    if st.button("Add to Profile", use_container_width=True):
        st.session_state.user_ratings[selected_mid] = user_rating
        st.toast(f"Added {movie_options[selected_mid]} ({user_rating}★) to your profile!", icon="🍿")
        
    st.write("---")
    
    # 2. View Active Profile
    st.markdown('<div class="sidebar-header">🍿 Rated Movies</div>', unsafe_allow_html=True)
    
    if not st.session_state.user_ratings:
        st.markdown('<div class="profile-empty">No movies rated yet. Search above to start!</div>', unsafe_allow_html=True)
    else:
        # Clear Profile button
        if st.button("Reset Profile", type="secondary", use_container_width=True):
            st.session_state.user_ratings = {}
            st.toast("Profile reset!", icon="🧹")
            st.rerun()
            
        # List of rated movies with deletion option
        for mid, rating in list(st.session_state.user_ratings.items()):
            title = recommender.id_to_title.get(mid, f"Movie ID {mid}")
            col_text, col_del = st.columns([0.8, 0.2])
            
            with col_text:
                st.markdown(f"**{title}**  \n⭐ {rating} / 5.0", unsafe_allow_html=True)
            with col_del:
                if st.button("❌", key=f"del_{mid}", help="Remove rating"):
                    del st.session_state.user_ratings[mid]
                    st.rerun()
            st.write("")
# --- MAIN SCREEN ---
# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">CineMatch</div>
    <div class="hero-subtitle">Premium hybrid recommendation system utilizing TF-IDF and collaborative filtering</div>
</div>
""", unsafe_allow_html=True)
# Main Navigation Tabs
tab_personal, tab_finder, tab_discover = st.tabs([
    "✨ Personalized For You", 
    "🔍 Similar Movies Finder", 
    "🔥 Discover & Trends"
])
# 1. TAB: Personalized Recommendations
with tab_personal:
    if not st.session_state.user_ratings:
        st.markdown("### Welcome to CineMatch! 🍿")
        st.info("Rate at least 2 or 3 movies in the sidebar to generate custom recommendations based on your unique tastes.")
        
        st.write("#### Get Started! Rate these popular movies:")
        popular_movies = recommender.get_popular_movies(top_n=6)
        
        # Render a quick rating layout for popular movies
        cols = st.columns(3)
        for i, row in popular_movies.iterrows():
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"""
                <div class="movie-card">
                    <div>
                        <div class="movie-title">{row['title']}</div>
                        <div class="genre-container">
                            {' '.join([f'<span class="genre-pill">{g}</span>' for g in row['genres'].split('|')])}
                        </div>
                    </div>
                    <div class="movie-metrics">
                        <span>⭐ {row['avg_rating']} / 5</span>
                        <span>👥 {row['rating_count']} ratings</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Button to quickly rate 5 stars
                if st.button(f"Rate 5★ - {row['title']}", key=f"quick_5_{row['movieId']}"):
                    st.session_state.user_ratings[row['movieId']] = 5.0
                    st.toast(f"Rated {row['title']} 5.0★!", icon="⭐")
                    st.rerun()
    else:
        st.subheader("Your Personalized Recommendations")
        
        # Mode selector for personalized filter
        algo_mode = st.radio(
            "Choose Recommendation Mode:",
            ["Hybrid (Recommended)", "Collaborative Filtering Only"],
            horizontal=True,
            help="Hybrid combines movie content (genres/tags) with collaborative user similarities to deliver highly tailored picks."
        )
        
        # Genre filter
        genres = recommender.get_genres()
        genre_filter = st.selectbox(
            "Filter by Genre (Optional):",
            ["All"] + genres,
            key="personal_genre"
        )
        genre_val = None if genre_filter == "All" else genre_filter
        
        st.write("")
        
        with st.spinner("Generating recommendations..."):
            if algo_mode.startswith("Hybrid"):
                recs = recommender.get_hybrid_recommendations(
                    st.session_state.user_ratings, top_n=9, genre_filter=genre_val
                )
            else:
                recs = recommender.get_collaborative_recommendations(
                    st.session_state.user_ratings, top_n=9, genre_filter=genre_val
                )
                
        if recs.empty:
            st.warning("No recommendations found. Try adjusting your genre filters or rating more movies!")
        else:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(recs.iterrows()):
                col_idx = idx % 3
                
                # Format scores based on recommender output
                score_html = ""
                if "hybrid_score" in row:
                    score_html = f'<div class="score-tag score-hybrid">Hybrid Score: {row["hybrid_score"]:.2f}</div>'
                elif "predicted_rating" in row and row["predicted_rating"] > 0:
                    score_html = f'<div class="score-tag score-collab">Predicted: {row["predicted_rating"]:.1f}★</div>'
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div>
                            <div class="movie-title">{row['title']}</div>
                            <div class="genre-container">
                                {' '.join([f'<span class="genre-pill">{g}</span>' for g in row['genres'].split('|')])}
                            </div>
                        </div>
                        <div>
                            {score_html}
                            <div class="movie-metrics">
                                <span>⭐ {row['avg_rating']} / 5</span>
                                <span>👥 {row['rating_count']} ratings</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
# 2. TAB: Similar Movies Finder
with tab_finder:
    st.subheader("Find Similar Movies")
    st.write("Select a movie from the catalog to find content-similar movies based on genres, descriptions, and user tags.")
    
    # Selection box
    search_movie_id = st.selectbox(
        "Select Movie:",
        options=list(movie_options.keys()),
        format_func=lambda x: movie_options[x],
        key="search_finder"
    )
    
    # Finder Genre Filter
    finder_genre_filter = st.selectbox(
        "Filter Similar Movies by Genre (Optional):",
        ["All"] + genres,
        key="finder_genre"
    )
    finder_genre_val = None if finder_genre_filter == "All" else finder_genre_filter
    
    if search_movie_id:
        target_movie = recommender.movies_df[recommender.movies_df["movieId"] == search_movie_id].iloc[0]
        
        # Display main movie details
        st.write("")
        st.markdown(f"### Recommending movies similar to: **{target_movie['title']}**")
        st.markdown(f"**Genres:** {target_movie['genres'].replace('|', '  |  ')} &nbsp;&nbsp;&nbsp;&nbsp; **Avg Rating:** ⭐ {target_movie['avg_rating']} &nbsp;&nbsp;&nbsp;&nbsp; **Total Ratings:** 👥 {target_movie['rating_count']}")
        st.write("---")
        
        with st.spinner("Finding matches..."):
            sim_recs = recommender.get_content_recommendations(
                search_movie_id, top_n=9, genre_filter=finder_genre_val
            )
            
        if sim_recs.empty:
            st.warning("No similar movies found. Try relaxing the genre filter.")
        else:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(sim_recs.iterrows()):
                col_idx = idx % 3
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div>
                            <div class="movie-title">{row['title']}</div>
                            <div class="genre-container">
                                {' '.join([f'<span class="genre-pill">{g}</span>' for g in row['genres'].split('|')])}
                            </div>
                        </div>
                        <div>
                            <div class="score-tag score-content">Similarity: {row['similarity'] * 100:.1f}%</div>
                            <div class="movie-metrics">
                                <span>⭐ {row['avg_rating']} / 5</span>
                                <span>👥 {row['rating_count']} ratings</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick add button to add this similar movie to user profile
                    if st.button("Rate this movie", key=f"rate_sim_{row['movieId']}"):
                        st.session_state.user_ratings[row['movieId']] = 4.0
                        st.toast(f"Added {row['title']} (4.0★) to your profile! Customize rating in the sidebar.", icon="🍿")
                        st.rerun()
# 3. TAB: Discover & Trends
with tab_discover:
    st.subheader("Trending & Popular Matches")
    st.write("Discover the overall highest-rated films in the catalog calculated using a Bayesian average to balance average scores with total rating volume.")
    
    # Discover Filters
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        discover_genre_filter = st.selectbox(
            "Select Genre:",
            ["All"] + genres,
            key="discover_genre"
        )
        discover_genre_val = None if discover_genre_filter == "All" else discover_genre_filter
        
    with col_d2:
        count_limit = st.slider(
            "Minimum Ratings Count:",
            min_value=1,
            max_value=100,
            value=10,
            key="discover_count"
        )
        
    st.write("")
    
    with st.spinner("Fetching catalog..."):
        # Get popular movies with genre filter
        trending = recommender.get_popular_movies(top_n=30, genre_filter=discover_genre_val)
        # Filter by minimum ratings
        trending = trending[trending["rating_count"] >= count_limit].head(12)
        
    if trending.empty:
        st.warning("No movies fit your search criteria. Try decreasing the minimum ratings count.")
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(trending.iterrows()):
            col_idx = idx % 3
            
            with cols[col_idx]:
                st.markdown(f"""
                <div class="movie-card">
                    <div>
                        <div class="movie-title">{row['title']}</div>
                        <div class="genre-container">
                            {' '.join([f'<span class="genre-pill">{g}</span>' for g in row['genres'].split('|')])}
                        </div>
                    </div>
                    <div>
                        <div class="movie-metrics">
                            <span>⭐ {row['avg_rating']} / 5</span>
                            <span>👥 {row['rating_count']} ratings</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick add button
                if st.button("Rate this movie", key=f"rate_disc_{row['movieId']}"):
                    st.session_state.user_ratings[row['movieId']] = 4.5
                    st.toast(f"Added {row['title']} (4.5★) to your profile!", icon="🍿")
                    st.rerun()
import os
import zipfile
import requests
import pandas as pd
import io
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
def download_movielens():
    print(f"Attempting to download MovieLens dataset from {MOVIELENS_URL}...")
    try:
        response = requests.get(MOVIELENS_URL, timeout=15)
        response.raise_for_status()
        
        # Extract zip directly from memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # The zip file contains a directory like ml-latest-small/
            # We want to extract the files inside it into DATA_DIR
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename:  # skip directories
                    continue
                
                # Extract to DATA_DIR
                source = zip_ref.open(member)
                target_path = os.path.join(DATA_DIR, filename)
                with open(target_path, "wb") as target:
                    target.write(source.read())
                print(f"Extracted: {filename} to {DATA_DIR}")
        print("MovieLens dataset downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False
def generate_fallback_data():
    print("Generating rich fallback mock dataset...")
    
    # 1. Curated list of popular movies across genres
    movies_data = [
        # Sci-Fi / Action
        (1, "The Matrix (1999)", "Action|Sci-Fi|Thriller"),
        (2, "Inception (2010)", "Action|Sci-Fi|Thriller"),
        (3, "Interstellar (2014)", "Sci-Fi|IMAX"),
        (4, "Star Wars: Episode IV - A New Hope (1977)", "Action|Adventure|Sci-Fi"),
        (5, "Star Wars: Episode V - The Empire Strikes Back (1980)", "Action|Adventure|Sci-Fi"),
        (6, "The Dark Knight (2008)", "Action|Crime|Drama|IMAX"),
        (7, "Avatar (2009)", "Action|Adventure|Sci-Fi|IMAX"),
        (8, "Blade Runner 2049 (2017)", "Sci-Fi|Thriller"),
        
        # Drama / Classics
        (9, "The Shawshank Redemption (1994)", "Crime|Drama"),
        (10, "The Godfather (1972)", "Crime|Drama"),
        (11, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller"),
        (12, "Forrest Gump (1994)", "Comedy|Drama|Romance"),
        (13, "Fight Club (1999)", "Action|Crime|Drama|Thriller"),
        (14, "Good Will Hunting (1997)", "Drama|Romance"),
        (15, "Schindler's List (1993)", "Drama|War"),
        
        # Romance / Comedy
        (16, "Titanic (1997)", "Drama|Romance"),
        (17, "La La Land (2016)", "Comedy|Drama|Musical|Romance"),
        (18, "The Notebook (2004)", "Drama|Romance"),
        (19, "Pride & Prejudice (2005)", "Drama|Romance"),
        (20, "500 Days of Summer (2009)", "Comedy|Drama|Romance"),
        (21, "Superbad (2007)", "Comedy"),
        (22, "The Hangover (2009)", "Comedy"),
        
        # Animation / Kids
        (23, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
        (24, "Spirited Away (2001)", "Adventure|Animation|Fantasy"),
        (25, "The Lion King (1994)", "Adventure|Animation|Children|Drama|Musical"),
        (26, "Finding Nemo (2003)", "Adventure|Animation|Children|Comedy"),
        (27, "Wall-E (2008)", "Adventure|Animation|Children|Romance|Sci-Fi"),
        (28, "Up (2009)", "Adventure|Animation|Children|Comedy|Drama"),
        
        # Thriller / Mystery / Horror
        (29, "The Silence of the Lambs (1991)", "Crime|Horror|Thriller"),
        (30, "Se7en (1995)", "Mystery|Thriller"),
        (31, "Shutter Island (2010)", "Drama|Mystery|Thriller"),
        (32, "The Prestige (2006)", "Drama|Mystery|Sci-Fi|Thriller"),
        (33, "Get Out (2017)", "Mystery|Psychological|Thriller"),
        (34, "The Shining (1980)", "Horror"),
        
        # Adventure / Fantasy
        (35, "The Lord of the Rings: The Fellowship of the Ring (2001)", "Adventure|Fantasy"),
        (36, "The Lord of the Rings: The Return of the King (2003)", "Adventure|Fantasy"),
        (37, "Harry Potter and the Sorcerer's Stone (2001)", "Adventure|Children|Fantasy"),
        (38, "Gladiator (2000)", "Action|Adventure|Drama"),
        (39, "Jurassic Park (1993)", "Action|Adventure|Sci-Fi|Thriller")
    ]
    
    movies_df = pd.DataFrame(movies_data, columns=["movieId", "title", "genres"])
    movies_df.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)
    
    # 2. Generate systematic ratings for mock users to create realistic collaborative filter results
    # User 1-3: Sci-Fi / Action fans
    # User 4-6: Drama / Classics fans
    # User 7-9: Romance / Comedy fans
    # User 10-12: Animation / Kids fans
    # User 13-15: Thriller / Mystery fans
    
    ratings_data = []
    
    # Sci-Fi / Action fans (like movies 1-8, dislike romance/comedy)
    for u in [1, 2, 3]:
        # highly rate sci-fi
        for m in [1, 2, 3, 4, 5, 6, 7, 8]:
            ratings_data.append((u, m, 4.5 if u % 2 == 0 else 5.0, 1260759000))
        # low rate romance
        for m in [16, 18, 20]:
            ratings_data.append((u, m, 1.5 if u % 2 == 0 else 2.0, 1260759000))
            
    # Drama fans (like movies 9-15, 38)
    for u in [4, 5, 6]:
        for m in [9, 10, 11, 12, 13, 14, 15, 38]:
            ratings_data.append((u, m, 4.5 if u % 2 == 0 else 5.0, 1260759000))
        for m in [21, 22, 34]:
            ratings_data.append((u, m, 2.0 if u % 2 == 0 else 2.5, 1260759000))
            
    # Romance / Comedy fans (like movies 12, 16-22)
    for u in [7, 8, 9]:
        for m in [12, 16, 17, 18, 19, 20, 21, 22]:
            ratings_data.append((u, m, 4.0 if u % 2 == 0 else 5.0, 1260759000))
        for m in [1, 2, 29, 30]:
            ratings_data.append((u, m, 1.5 if u % 2 == 0 else 2.0, 1260759000))
            
    # Animation fans (like movies 23-28)
    for u in [10, 11, 12]:
        for m in [23, 24, 25, 26, 27, 28]:
            ratings_data.append((u, m, 4.5 if u % 2 == 0 else 5.0, 1260759000))
        for m in [29, 30, 34]:
            ratings_data.append((u, m, 1.0 if u % 2 == 0 else 2.0, 1260759000))
    # Thriller / Mystery fans (like movies 2, 6, 29-34)
    for u in [13, 14, 15]:
        for m in [2, 6, 29, 30, 31, 32, 33, 34]:
            ratings_data.append((u, m, 4.5 if u % 2 == 0 else 5.0, 1260759000))
        for m in [23, 25, 18]:
            ratings_data.append((u, m, 2.0 if u % 2 == 0 else 1.5, 1260759000))
            
    ratings_df = pd.DataFrame(ratings_data, columns=["userId", "movieId", "rating", "timestamp"])
    ratings_df.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
    
    # 3. Create mock tags
    tags_data = [
        (1, 1, "mind-bending", 1260759000),
        (1, 2, "dreams", 1260759000),
        (2, 3, "space-travel", 1260759000),
        (3, 6, "joker", 1260759000),
        (4, 9, "hope", 1260759000),
        (5, 10, "mafia", 1260759000),
        (7, 17, "musical", 1260759000),
        (10, 24, "studio-ghibli", 1260759000),
        (13, 29, "serial-killer", 1260759000),
        (15, 32, "magic", 1260759000),
    ]
    tags_df = pd.DataFrame(tags_data, columns=["userId", "movieId", "tag", "timestamp"])
    tags_df.to_csv(os.path.join(DATA_DIR, "tags.csv"), index=False)
    
    # 4. Create mock links
    links_data = [(m[0], m[0] * 1000, m[0] * 2000) for m in movies_data]
    links_df = pd.DataFrame(links_data, columns=["movieId", "imdbId", "tmdbId"])
    links_df.to_csv(os.path.join(DATA_DIR, "links.csv"), index=False)
    
    print("Fallback mock dataset successfully generated in data/")
def main():
    setup_directories()
    # Try downloading first, otherwise fall back to generating local mock data
    if not download_movielens():
        generate_fallback_data()
if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
class MovieRecommender:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default to the local data folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "data")
        
        self.data_dir = data_dir
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        
        # Models / matrices
        self.tfidf_matrix = None
        self.vectorizer = None
        self.user_item_matrix = None
        
        self.load_data()
        self.preprocess()
        self.fit_content_model()
        self.fit_collaborative_model()
    def load_data(self):
        """Loads MovieLens datasets from CSV files."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        tags_path = os.path.join(self.data_dir, "tags.csv")
        
        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"Required data files not found in {self.data_dir}. "
                "Please run download_data.py first."
            )
            
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        if os.path.exists(tags_path):
            self.tags_df = pd.read_csv(tags_path)
        else:
            self.tags_df = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
            
        print(f"Loaded {len(self.movies_df)} movies, {len(self.ratings_df)} ratings, and {len(self.tags_df)} tags.")
    def preprocess(self):
        """Cleans and structures the data for modeling."""
        # Calculate average rating and number of ratings for each movie
        rating_stats = self.ratings_df.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        ).reset_index()
        
        # Merge stats into movies_df
        self.movies_df = pd.merge(self.movies_df, rating_stats, on="movieId", how="left")
        self.movies_df["avg_rating"] = self.movies_df["avg_rating"].fillna(0.0).round(2)
        self.movies_df["rating_count"] = self.movies_df["rating_count"].fillna(0).astype(int)
        
        # Parse genres into a space-separated string for TF-IDF (replace '|' with ' ')
        self.movies_df["genres_clean"] = self.movies_df["genres"].fillna("").str.replace("|", " ", regex=False)
        
        # Combine user tags per movie
        if not self.tags_df.empty:
            movie_tags = self.tags_df.groupby("movieId")["tag"].apply(
                lambda x: " ".join(str(tag) for tag in x.dropna())
            ).reset_index()
            movie_tags.rename(columns={"tag": "tags_combined"}, inplace=True)
            self.movies_df = pd.merge(self.movies_df, movie_tags, on="movieId", how="left")
        else:
            self.movies_df["tags_combined"] = ""
            
        self.movies_df["tags_combined"] = self.movies_df["tags_combined"].fillna("")
        
        # Create a combined metadata column (title + genres + tags)
        self.movies_df["metadata"] = (
            self.movies_df["title"] + " " + 
            self.movies_df["genres_clean"] + " " + 
            self.movies_df["tags_combined"]
        ).str.lower()
        
        # Quick access dictionaries
        self.title_to_id = dict(zip(self.movies_df["title"], self.movies_df["movieId"]))
        self.id_to_title = dict(zip(self.movies_df["movieId"], self.movies_df["title"]))
    def fit_content_model(self):
        """Creates TF-IDF matrix for content-based filtering."""
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["metadata"])
        print("TF-IDF matrix built for Content-Based filtering.")
    def fit_collaborative_model(self):
        """Builds the User-Item matrix for collaborative filtering."""
        # Index: userId, Columns: movieId, Values: rating
        self.user_item_matrix = self.ratings_df.pivot(
            index="userId", columns="movieId", values="rating"
        )
        print(f"User-Item Matrix shape: {self.user_item_matrix.shape}")
    def get_genres(self):
        """Extracts unique genres present in the dataset."""
        unique_genres = set()
        for genres_str in self.movies_df["genres"].dropna():
            for g in genres_str.split("|"):
                if g and g != "(no genres listed)":
                    unique_genres.add(g)
        return sorted(list(unique_genres))
    def search_movies(self, query, top_n=10):
        """Searches movies by title (case-insensitive substring match)."""
        if not query:
            return pd.DataFrame()
        matches = self.movies_df[self.movies_df["title"].str.contains(query, case=False, na=False)]
        return matches.sort_values(by="rating_count", ascending=False).head(top_n)
    def get_content_recommendations(self, movie_id, top_n=10, genre_filter=None):
        """Recommends movies similar to a target movie using content similarity."""
        # Find index of target movie
        idx_matches = self.movies_df.index[self.movies_df["movieId"] == movie_id].tolist()
        if not idx_matches:
            return pd.DataFrame()
        idx = idx_matches[0]
        
        # Compute cosine similarity of target movie with all others
        target_vector = self.tfidf_matrix[idx]
        sim_scores = linear_kernel(target_vector, self.tfidf_matrix).flatten()
        
        # Create dataframe of similarity scores
        sim_df = pd.DataFrame({
            "index": range(len(sim_scores)),
            "similarity": sim_scores
        })
        
        # Exclude target movie itself
        sim_df = sim_df[sim_df["index"] != idx]
        
        # Merge with movies_df to get details
        recommendations = pd.merge(sim_df, self.movies_df.reset_index(), on="index")
        
        # Apply genre filter if specified
        if genre_filter:
            recommendations = recommendations[
                recommendations["genres"].str.contains(genre_filter, case=False, na=False)
            ]
            
        # Sort by similarity and return top N
        recommendations = recommendations.sort_values(by="similarity", ascending=False)
        return recommendations.head(top_n)[
            ["movieId", "title", "genres", "avg_rating", "rating_count", "similarity"]
        ]
    def get_collaborative_recommendations(self, new_user_ratings, top_n=10, genre_filter=None):
        """
        Recommends movies using User-Based Collaborative Filtering.
        new_user_ratings: dict mapping movieId (int) to rating (float)
        """
        if not new_user_ratings:
            # Cold-start: return top rated popular movies
            return self.get_popular_movies(top_n=top_n, genre_filter=genre_filter)
            
        # 1. Create a Series for the new user aligned with the user-item matrix columns
        new_user_series = pd.Series(index=self.user_item_matrix.columns, dtype=float)
        for mid, rating in new_user_ratings.items():
            if mid in new_user_series.index:
                new_user_series[mid] = rating
                
        # 2. Compute cosine similarity between the new user and all existing users
        # Fill NaNs with 0 for vector calculation
        existing_users_filled = self.user_item_matrix.fillna(0)
        new_user_filled = new_user_series.fillna(0).values.reshape(1, -1)
        
        similarities = cosine_similarity(new_user_filled, existing_users_filled).flatten()
        
        # Create similarity dataframe
        sim_users_df = pd.DataFrame({
            "userId": self.user_item_matrix.index,
            "similarity": similarities
        }).sort_values(by="similarity", ascending=False)
        
        # Keep users with positive similarity
        sim_users_df = sim_users_df[sim_users_df["similarity"] > 0.05]
        
        if sim_users_df.empty:
            # Fall back to content-based or popularity-based
            return self.get_popular_movies(top_n=top_n, genre_filter=genre_filter)
            
        # Keep top K similar users to prevent noise
        top_k_users = sim_users_df.head(30)
        
        # 3. Predict ratings for candidate movies (movies the new user hasn't rated yet)
        candidate_movie_ids = [mid for mid in self.user_item_matrix.columns if mid not in new_user_ratings]
        
        # Slice user-item matrix to only contain top K similar users and candidate movies
        sub_matrix = self.user_item_matrix.loc[top_k_users["userId"], candidate_movie_ids]
        
        # Calculate weighted average ratings
        # R_pred = sum(similarity * rating) / sum(similarity)
        sim_weights = top_k_users.set_index("userId")["similarity"]
        
        weighted_ratings = sub_matrix.mul(sim_weights, axis=0)
        sum_weighted_ratings = weighted_ratings.sum(axis=0, min_count=1) # min_count=1 returns NaN if all are NaN
        sum_similarities = sub_matrix.notna().mul(sim_weights, axis=0).sum(axis=0)
        
        predicted_ratings = (sum_weighted_ratings / sum_similarities).fillna(0)
        
        # 4. Compile predictions
        pred_df = pd.DataFrame({
            "movieId": predicted_ratings.index,
            "predicted_rating": predicted_ratings.values
        })
        
        # Merge movie info
        recommendations = pd.merge(pred_df, self.movies_df, on="movieId")
        
        # Apply genre filter
        if genre_filter:
            recommendations = recommendations[
                recommendations["genres"].str.contains(genre_filter, case=False, na=False)
            ]
            
        # Filter out movies with low overall ratings or counts to keep quality high
        # Unless the dataset is tiny (mock data)
        min_rating_count = 5 if len(self.ratings_df) > 1000 else 1
        recommendations = recommendations[recommendations["rating_count"] >= min_rating_count]
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values(by="predicted_rating", ascending=False)
        return recommendations.head(top_n)[
            ["movieId", "title", "genres", "avg_rating", "rating_count", "predicted_rating"]
        ]
    def get_hybrid_recommendations(self, new_user_ratings, top_n=10, genre_filter=None):
        """
        Hybrid recommender:
        1. Generates collaborative filtering candidates.
        2. Computes content-based similarity to the user's highly-rated movies (rating >= 4.0).
        3. Combines collaborative predicted ratings and content similarity to re-rank.
        """
        if not new_user_ratings:
            return self.get_popular_movies(top_n=top_n, genre_filter=genre_filter)
            
        # Get collaborative candidates (request more candidates to allow reranking)
        collab_recs = self.get_collaborative_recommendations(new_user_ratings, top_n=top_n * 3, genre_filter=genre_filter)
        if collab_recs.empty:
            return self.get_popular_movies(top_n=top_n, genre_filter=genre_filter)
            
        # Find user's highly rated movies (e.g. 4.0 and above)
        highly_rated_mids = [mid for mid, rating in new_user_ratings.items() if rating >= 4.0]
        
        if not highly_rated_mids:
            # If no highly rated movies, just return collaborative filtering directly
            return collab_recs.head(top_n)
            
        # For each candidate, compute average similarity to the user's highly rated movies
        candidate_ids = collab_recs["movieId"].tolist()
        
        # Get TF-IDF vectors for candidates and highly rated movies
        candidate_indices = [self.movies_df.index[self.movies_df["movieId"] == mid][0] for mid in candidate_ids]
        highly_rated_indices = [self.movies_df.index[self.movies_df["movieId"] == mid][0] for mid in highly_rated_mids]
        
        candidate_vectors = self.tfidf_matrix[candidate_indices]
        highly_rated_vectors = self.tfidf_matrix[highly_rated_indices]
        
        # Calculate similarity matrix between candidates and highly rated items
        # Shape: (len(candidate_indices), len(highly_rated_indices))
        item_sim_matrix = cosine_similarity(candidate_vectors, highly_rated_vectors)
        
        # Take the maximum similarity (or mean similarity) of a candidate to ANY of the user's favorite movies
        max_content_sim = item_sim_matrix.max(axis=1)
        
        # Add content similarity to collaborative dataframe
        collab_recs["content_affinity"] = max_content_sim
        
        # Normalize predicted ratings to 0-1 scale (within the candidates)
        min_pred = collab_recs["predicted_rating"].min()
        max_pred = collab_recs["predicted_rating"].max()
        pred_range = max_pred - min_pred if max_pred != min_pred else 1.0
        collab_recs["normalized_predicted"] = (collab_recs["predicted_rating"] - min_pred) / pred_range
        
        # Combine normalized predicted rating and content affinity
        # 60% collaborative filtering score, 40% content similarity to user favorites
        collab_recs["hybrid_score"] = (
            0.6 * collab_recs["normalized_predicted"] + 
            0.4 * collab_recs["content_affinity"]
        )
        
        # Re-sort by hybrid score
        hybrid_recs = collab_recs.sort_values(by="hybrid_score", ascending=False)
        return hybrid_recs.head(top_n)[
            ["movieId", "title", "genres", "avg_rating", "rating_count", "predicted_rating", "content_affinity", "hybrid_score"]
        ]
    def get_popular_movies(self, top_n=10, genre_filter=None):
        """Returns popular movies based on rating counts and average rating (Bayesian average)."""
        # Bayesian average: (v*R + m*C) / (v+m)
        # where v is rating count, R is average rating, C is mean rating across all movies,
        # and m is minimum rating count threshold.
        C = self.ratings_df["rating"].mean()
        m = 10 if len(self.ratings_df) > 1000 else 1
        
        df = self.movies_df.copy()
        
        # Calculate bayesian average
        df["bayesian_score"] = (
            (df["rating_count"] * df["avg_rating"]) + (m * C)
        ) / (df["rating_count"] + m)
        
        if genre_filter:
            df = df[df["genres"].str.contains(genre_filter, case=False, na=False)]
            
        popular = df.sort_values(by="bayesian_score", ascending=False).head(top_n)
        return popular[["movieId", "title", "genres", "avg_rating", "rating_count"]]
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
import sys
from recommender import MovieRecommender
def main():
    print("Initializing MovieRecommender...")
    try:
        recommender = MovieRecommender()
    except Exception as e:
        print(f"FAILED to initialize recommender: {e}")
        sys.exit(1)
        
    print("\n--- Testing Search ---")
    search_query = "Matrix"
    results = recommender.search_movies(search_query)
    if not results.empty:
        print(f"Found {len(results)} matches for '{search_query}'. Top match:")
        print(results.iloc[0][["movieId", "title", "genres"]])
    else:
        print(f"No matches found for '{search_query}'.")
        
    # Get a movie ID to test content recommendations
    # We will look for "The Matrix (1999)" or the first movie in the dataset
    movie_id = None
    if not results.empty:
        movie_id = results.iloc[0]["movieId"]
    else:
        movie_id = recommender.movies_df.iloc[0]["movieId"]
        
    movie_title = recommender.id_to_title[movie_id]
    print(f"\n--- Testing Content Recommendations for movie: '{movie_title}' (ID: {movie_id}) ---")
    content_recs = recommender.get_content_recommendations(movie_id, top_n=5)
    print(content_recs)
    
    print("\n--- Testing Genres List ---")
    genres = recommender.get_genres()
    print("Unique genres:", genres[:10], "... total:", len(genres))
    
    print("\n--- Testing Collaborative Filtering for a mock user ---")
    # Simulate a user who likes Matrix (movieId from search) and Toy Story (if exists)
    toy_story_results = recommender.search_movies("Toy Story")
    mock_ratings = {}
    if movie_id:
        mock_ratings[movie_id] = 5.0 # likes Matrix
    if not toy_story_results.empty:
        toy_story_id = toy_story_results.iloc[0]["movieId"]
        mock_ratings[toy_story_id] = 4.5 # likes Toy Story
        
    print(f"Mock user ratings: {mock_ratings}")
    collab_recs = recommender.get_collaborative_recommendations(mock_ratings, top_n=5)
    print(collab_recs)
    
    print("\n--- Testing Hybrid Recommendations for mock user ---")
    hybrid_recs = recommender.get_hybrid_recommendations(mock_ratings, top_n=5)
    print(hybrid_recs)
    
    print("\nAll recommendation engine tests completed successfully!")
if __name__ == "__main__":
    main()
