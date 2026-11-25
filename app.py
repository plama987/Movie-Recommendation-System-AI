import os
from datetime import datetime

import pandas as pd
import streamlit as st

from recommender import MovieRecommender


DATA_PATH = "movies.csv"
FEEDBACK_PATH = "feedback/user_feedback.csv"


st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


def save_feedback(movie_title, selected_movie, reaction):
    """Append a user's reaction to a local CSV file."""
    os.makedirs("feedback", exist_ok=True)
    row = pd.DataFrame([{
        "selected_movie": selected_movie,
        "recommended_movie": movie_title,
        "feedback": reaction,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }])
    row.to_csv(
        FEEDBACK_PATH,
        mode="a",
        header=not os.path.exists(FEEDBACK_PATH),
        index=False
    )


@st.cache_resource
def load_model():
    return MovieRecommender(DATA_PATH)


st.markdown(
    """
    <style>
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {color: #9aa4b2; margin-top: .2rem;}
    .movie-card {padding: 1rem; border: 1px solid rgba(128,128,128,.25);
                 border-radius: 12px; margin-bottom: .7rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🎬 CineMatch</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Discover your next favorite movie with AI-powered recommendations.</p>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("About CineMatch")
    st.write(
        "CineMatch uses content-based filtering. It compares movie genres, "
        "keywords, plot summaries, cast, and director information."
    )
    st.divider()
    st.caption("Built with Python, Streamlit and scikit-learn.")
    st.caption("Dataset: uploaded TMDB movie metadata CSV.")

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}. Place movies.csv in the data folder.")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the dataset or recommendation model: {exc}")
    st.stop()

movie_titles = model.get_movie_list()
genres = model.get_genres()

st.subheader("Find movies you'll enjoy")
left, right = st.columns([2, 1])

with left:
    selected_movie = st.selectbox(
        "Choose a movie you already like",
        movie_titles,
        index=0
    )

with right:
    selected_genre = st.selectbox(
        "Optional genre filter",
        ["All genres"] + genres
    )

count = st.slider("Number of recommendations", 5, 20, 10)

if st.button("✨ Recommend movies", type="primary", use_container_width=True):
    genre_filter = None if selected_genre == "All genres" else selected_genre
    recommendations = model.recommend_movies(
        selected_movie=selected_movie,
        selected_genre=genre_filter,
        limit=count
    )
    st.session_state["recommendations"] = recommendations
    st.session_state["selected_movie"] = selected_movie

if "recommendations" in st.session_state:
    results = st.session_state["recommendations"]
    source_movie = st.session_state["selected_movie"]

    st.divider()
    st.subheader(f"Because you liked: {source_movie}")

    if results.empty:
        st.warning("No recommendations matched this filter. Try another genre or movie.")
    else:
        st.caption(
            "Similarity is a text-feature similarity score, not a prediction of your personal rating."
        )
        for number, (_, movie) in enumerate(results.iterrows(), start=1):
            with st.container(border=True):
                title_col, score_col = st.columns([4, 1])
                with title_col:
                    year = movie["year"]
                    st.markdown(f"### {number}. {movie['title']} ({year})")
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(movie["overview"])
                    st.caption(
                        f"TMDB dataset rating: {movie['rating']}  |  "
                        f"Popularity: {movie['popularity']}"
                    )
                with score_col:
                    st.metric("Similarity", f"{movie['similarity']}%")
                    st.progress(min(float(movie["similarity"]) / 100, 1.0))

                feedback_col1, feedback_col2, _ = st.columns([1, 1, 5])
                with feedback_col1:
                    if st.button("👍 Like", key=f"like_{movie['movie_id']}"):
                        save_feedback(movie["title"], source_movie, "Like")
                        st.success("Thanks! Your feedback was saved.")
                with feedback_col2:
                    if st.button("👎 Dislike", key=f"dislike_{movie['movie_id']}"):
                        save_feedback(movie["title"], source_movie, "Dislike")
                        st.success("Your feedback was saved.")

with st.expander("How does the recommendation engine work?"):
    st.markdown(
        """
        1. **Metadata preparation:** The application reads movie genres, keywords,
           plot summaries, cast, and director information.
        2. **TF-IDF:** Text is converted into numerical feature vectors. Words
           that are informative for a movie receive more weight.
        3. **Cosine similarity:** The system compares the selected movie's vector
           with the vectors of other movies.
        4. **Ranking:** Movies with the highest similarity scores are shown first.
        5. **HCI:** Dropdowns, a genre filter, adjustable result count, readable
           cards, and feedback buttons make the system interactive.
        """
    )

st.caption(
    "Academic demonstration project. Feedback is stored locally and does not "
    "currently retrain the recommendation model."
)
