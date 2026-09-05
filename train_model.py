import pandas as pd
import ast
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# Rename movie_id column
credits = credits.rename(columns={"movie_id": "id"})


# Merge datasets
movies = movies.merge(credits, on="id")


# Select required columns
movies = movies[
    ["id", "title_x", "overview", "genres", "keywords", "cast", "crew"]
]

movies = movies.rename(columns={"title_x": "title"})


# Convert JSON-like strings into useful text
def convert(data):
    try:
        data = ast.literal_eval(data)
        return " ".join([item["name"] for item in data])
    except:
        return ""


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


# Extract top 3 cast members
def get_cast(data):
    try:
        data = ast.literal_eval(data)
        return " ".join([item["name"] for item in data[:3]])
    except:
        return ""


movies["cast"] = movies["cast"].apply(get_cast)


# Extract director
def get_director(data):
    try:
        data = ast.literal_eval(data)

        for item in data:
            if item["job"] == "Director":
                return item["name"]

        return ""

    except:
        return ""


movies["crew"] = movies["crew"].apply(get_director)


# Fill missing values
movies["overview"] = movies["overview"].fillna("")


# Combine all features
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["crew"]
)


# TF-IDF vectorization
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

vectors = tfidf.fit_transform(movies["tags"])


# Calculate cosine similarity
similarity = cosine_similarity(vectors)


# Save required data
pickle.dump(movies, open("movie_data.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Model trained successfully!")
print("Number of movies:", len(movies))