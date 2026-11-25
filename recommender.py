import ast
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_names(value):
    """Extract readable names from TMDB-style list/dictionary text."""
    if pd.isna(value) or not str(value).strip():
        return ""

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            names = []
            for item in parsed:
                if isinstance(item, dict) and item.get("name"):
                    names.append(str(item["name"]))
                elif isinstance(item, str):
                    names.append(item)
            return " ".join(names)
        if isinstance(parsed, dict):
            return str(parsed.get("name", ""))
    except (ValueError, SyntaxError):
        pass

    # Fallback for imperfect metadata strings.
    return " ".join(re.findall(r"'name':\s*'([^']+)'", text))


class MovieRecommender:
    """Content-based movie recommender built from TMDB movie metadata."""

    def __init__(self, dataset_path):
        self.movies = pd.read_csv(dataset_path)

        required = {"title", "genres"}
        missing = required - set(self.movies.columns)
        if missing:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(sorted(missing))
            )

        self.movies = self.movies.dropna(subset=["title"]).copy()
        self.movies["title"] = self.movies["title"].astype(str)
        self.movies = self.movies.drop_duplicates(subset=["title"]).reset_index(drop=True)

        # Convert structured TMDB metadata into searchable text.
        for column in ["genres", "keywords", "cast", "production_companies"]:
            if column in self.movies.columns:
                self.movies[column + "_text"] = self.movies[column].apply(extract_names)
            else:
                self.movies[column + "_text"] = ""

        for column in ["overview", "tagline", "director"]:
            if column not in self.movies.columns:
                self.movies[column] = ""
            self.movies[column] = self.movies[column].fillna("").astype(str)

        self.movies["release_year"] = ""
        if "release_date" in self.movies.columns:
            self.movies["release_year"] = (
                pd.to_datetime(self.movies["release_date"], errors="coerce")
                .dt.year.fillna("").astype(str)
            )

        # Give genres, overview, cast and director a combined representation.
        self.movies["features"] = (
            self.movies["genres_text"] + " " +
            self.movies["keywords_text"] + " " +
            self.movies["overview"] + " " +
            self.movies["tagline"] + " " +
            self.movies["cast_text"] + " " +
            self.movies["director"]
        ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=12000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["features"])

    def get_movie_list(self):
        return sorted(self.movies["title"].dropna().unique().tolist())

    def get_genres(self):
        genres = set()
        for value in self.movies["genres_text"]:
            genres.update(g.strip() for g in value.split() if g.strip())
        return sorted(genres)

    def recommend_movies(self, selected_movie, selected_genre=None, limit=10):
        matches = self.movies.index[self.movies["title"] == selected_movie].tolist()
        if not matches:
            return pd.DataFrame()

        movie_index = matches[0]
        scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix
        ).flatten()

        ranked_indices = scores.argsort()[::-1]
        results = []

        for idx in ranked_indices:
            if idx == movie_index:
                continue

            movie = self.movies.iloc[idx]
            movie_genres = movie["genres_text"]

            if selected_genre and selected_genre not in movie_genres.split():
                continue

            results.append({
                "title": movie["title"],
                "year": movie["release_year"] or "N/A",
                "genres": movie_genres.replace(" ", ", ") or "Not listed",
                "overview": movie["overview"] or "No overview available.",
                "similarity": round(float(scores[idx]) * 100, 1),
                "rating": movie.get("vote_average", "N/A"),
                "popularity": movie.get("popularity", "N/A"),
                "movie_id": movie.get("id", idx)
            })

            if len(results) >= limit:
                break

        return pd.DataFrame(results)
