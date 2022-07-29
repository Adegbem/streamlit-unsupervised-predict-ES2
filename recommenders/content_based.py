"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv', sep=',')

# Instantiating and generating the count matrix
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

def data_preprocessing(df):
    # Subset of the data
    df = df[:27000]
    df['clean_title'] = df.title.apply(clean_title)
    df['genres'] = df['genres'].apply(lambda x: x.replace('|', ' '))
    df['concat'] = (pd.Series(df[['clean_title', 'genres']].values.tolist()).str.join(' '))
    return df

def content_model(movie_list, top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    # Convenient indexes to map between movie titles and indexes of the 'all_df' dataframe
    movies = data_preprocessing(movies_df)

    # Produce a feature matrix, where each row corresponds to a movie,
    # with TF-IDF features as columns
    tf_enrich_matrix = tf.fit_transform(movies['concat'])

    movies_str = []
    for title in movie_list:
        cl_title = clean_title(title)
        cl_genres = movies_df.loc[movies_df['title'] == title, 'genres']
        cl_genres = cl_genres.values[0].replace('|', ' ')
        movie_detail = cl_title + ' ' + cl_genres
        movies_str.append(movie_detail)
    titles = ' '.join(movies_str)

    title_comb = clean_title(titles)
    query_vec = tf.transform([title_comb])
    similarity = cosine_similarity(query_vec, tf_enrich_matrix).flatten()
    indices = np.argpartition(similarity, -5)[-15:]
    results = movies[['clean_title']].iloc[indices].iloc[::-1]
    results = results.clean_title.to_list()

    recommended_movies = []
    count = 0
    for title in results:
        movies_clean = [clean_title(x) for x in movie_list]
        if title in movies_clean:
            pass
        else:
            recommended_movies.append(title)
            count += 1
        if count == 10:
            break
    return recommended_movies
