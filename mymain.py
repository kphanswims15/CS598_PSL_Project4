import streamlit as st
from streamlit_star_rating import st_star_rating
import pandas as pd
import numpy as np
from PIL import Image

movies_titles = pd.read_csv('https://github.com/kphanswims15/CS598_PSL_Project4/blob/main/ml-1m/movies.dat', sep = '::', engine = 'python',
                        encoding = "ISO-8859-1", header = None)
movies_titles.columns = ['MovieID', 'Title', 'Genres']

def get_movie_details(movie):
    movie_title = movies_titles.loc[movies_titles['MovieID'] == int(movie[1:])]
    filename = f'MovieImages\\{movie[1:]}.jpg'
    image = Image.open(filename)
    title = movie_title['Title'].to_string()

    return image, title

# Load Data
@st.cache_data
def load_data():
    # Load the rating matrix and similarity matrix
    rating_matrix = pd.read_csv("https://github.com/kphanswims15/CS598_PSL_Project4/blob/main/I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv?raw=true", index_col=0)
    similarity_matrix = pd.read_csv("https://github.com/kphanswims15/CS598_PSL_Project4/blob/main/S_matrix.csv?raw=true", index_col=0)
    return rating_matrix, similarity_matrix

def myIBCF(new_user_ratings, similarity_matrix):
    predictions = {}
    for movie in similarity_matrix.index:
        if pd.isna(new_user_ratings.get(movie, np.nan)): 
            similar_movies = similarity_matrix[movie].dropna()
            rated_movies = new_user_ratings.dropna()
            relevant_movies = rated_movies.index.intersection(similar_movies.index)

            if len(relevant_movies) > 0:
                weights = similar_movies[relevant_movies]
                ratings = rated_movies[relevant_movies]
                denominator = weights.sum()
                numerator = (weights * ratings).sum()

                if denominator > 0:
                    predictions[movie] = numerator / denominator
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_predictions[:10]]

def main():
    st.title("Movie Recommender System")
    st.subheader("Rate Movies to Get Personalized Recommendations")

    rating_matrix, similarity_matrix = load_data()

    sample_movies = rating_matrix.columns[:100]
    st.write("Please rate the following sample movies (1-5 stars or leave blank):")
    user_ratings = {}
    for movie in sample_movies:
        image, title = get_movie_details(movie)

        st.image(image)
        user_ratings[movie] = st_star_rating(title[1:], maxValue=5, defaultValue=0, key=movie)

    user_ratings_series = pd.Series(user_ratings).dropna()

    if st.button("Get Recommendations"):
        if user_ratings_series.empty:
            st.warning("Please rate at least one movie to get recommendations.")
        else:
            recommendations = myIBCF(user_ratings_series, similarity_matrix)
            st.success("Top 10 Recommended Movies for You:")
            for movie in recommendations:
                image, title = get_movie_details(movie)

                st.image(image)
                st.write(title)

if __name__ == "__main__":
    main()