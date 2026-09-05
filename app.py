import webbrowser
from flask import Flask, render_template, request
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# --------------------------------------------------
# LOAD TRAINED DATA
# --------------------------------------------------

movies = pickle.load(open("movie_data.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))


# --------------------------------------------------
# FUNCTION 1: RECOMMEND FROM EXISTING MOVIES
# --------------------------------------------------

def recommend_existing(movie_name):

    matches = movies[
        movies["title"].str.lower() == movie_name.lower()
    ]

    if matches.empty:
        return []

    index = matches.index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i, score in movie_list:
        recommendations.append({
            "title": movies.iloc[i]["title"],
            "score": round(score * 100, 2)
        })

    return recommendations


# --------------------------------------------------
# FUNCTION 2: RECOMMEND FOR USER'S OWN MOVIE
# --------------------------------------------------

def recommend_custom_movie(title, genre, overview, keywords, cast, director):

    # Create text for the new movie
    custom_text = (
        str(overview) + " " +
        str(genre) + " " +
        str(keywords) + " " +
        str(cast) + " " +
        str(director)
    )

    # Existing movie data
    existing_text = movies["tags"].fillna("").tolist()

    # Add custom movie to existing movies temporarily
    all_text = existing_text + [custom_text]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    vectors = vectorizer.fit_transform(all_text)

    # Custom movie is the last vector
    custom_vector = vectors[-1]

    # Compare custom movie with existing movies
    similarity_scores = cosine_similarity(
        custom_vector,
        vectors[:-1]
    )[0]

    # Get top 5 similar movies
    movie_list = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )[:5]

    recommendations = []

    for i, score in movie_list:
        recommendations.append({
            "title": movies.iloc[i]["title"],
            "score": round(score * 100, 2)
        })

    return recommendations


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    movie_titles = sorted(
        movies["title"].dropna().unique().tolist()
    )

    return render_template(
        "index.html",
        movies=movie_titles
    )


# --------------------------------------------------
# EXISTING MOVIE RECOMMENDATION
# --------------------------------------------------

@app.route("/recommend_existing", methods=["POST"])
def recommend_existing_route():

    movie_name = request.form.get("movie")

    recommendations = recommend_existing(movie_name)

    return render_template(
        "index.html",
        movies=sorted(movies["title"].dropna().unique().tolist()),
        recommendations=recommendations,
        selected_movie=movie_name,
        mode="existing"
    )


# --------------------------------------------------
# CUSTOM MOVIE RECOMMENDATION
# --------------------------------------------------

@app.route("/recommend_custom", methods=["POST"])
def recommend_custom_route():

    title = request.form.get("title")
    genre = request.form.get("genre")
    overview = request.form.get("overview")
    keywords = request.form.get("keywords")
    cast = request.form.get("cast")
    director = request.form.get("director")

    recommendations = recommend_custom_movie(
        title,
        genre,
        overview,
        keywords,
        cast,
        director
    )

    return render_template(
        "index.html",
        movies=sorted(movies["title"].dropna().unique().tolist()),
        recommendations=recommendations,
        custom_title=title,
        mode="custom"
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)