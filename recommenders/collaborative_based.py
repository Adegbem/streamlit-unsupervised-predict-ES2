"""
    Collaborative-based filtering for item recommendation.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.
    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.
    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!
    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.
    ---------------------------------------------------------------------
    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.
"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from numpy import random

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1, inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model = pickle.load(open('resources/models/baselineOnly.pkl', 'rb'))


def prediction_item(movie_list):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.
    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.
    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.
    """
    # Data preprosessing
    reader = Reader()
    load_df = Dataset.load_from_df(ratings_df, reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    id_store = []
    for movie in movie_list:
        iid = movies_df.loc[movies_df['title'] == movie, 'movieId'].values[0]
        for ui in a_train.all_users():
            predictions.append(model.predict(iid=iid, uid=ui, verbose=False))
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:20]:
            id_store.append(pred.uid)
    return id_store


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def collab_model(movie_list, top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
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
    pred_users = prediction_item(movie_list)
    pred_df = ratings_df[ratings_df['userId'].isin(pred_users)]
    pred_mod_df = pred_df.merge(movies_df, on='movieId')
    pred_mod_df = pred_mod_df.groupby('title').agg(rating_count=('rating', 'count'),
                                                   rating_mean=('rating', 'mean')).reset_index()
    # Attempt to create the formula for weighted rating
    vote_counts = pred_mod_df[pred_mod_df['rating_count'].notnull()]['rating_count']
    vote_averages = pred_mod_df[pred_mod_df['rating_mean'].notnull()]['rating_mean']
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)
    # creating a function to calculate the weighting rating as recommended by IMDB
    def weighted_rating(x, mini=m, avg_votes=C):
        votes = x['rating_count']
        avg_rating = x['rating_mean']
        return (votes/(votes+mini) * avg_rating) + (mini/(mini+votes) * avg_votes)

    # Creating the feature 'weighted_rating'
    pred_mod_df['weighted_rating'] = pred_mod_df.apply(weighted_rating, axis=1)

    # Convenient indexes to map between movie titles and indexes of the 'all_df' dataframe
    titles = pred_mod_df['title']
    indices = pd.Series(pred_mod_df.index, index=pred_mod_df['title'])
    df_array = np.array(pred_mod_df[["rating_count", "rating_mean", "weighted_rating"]])
    cosine_sim_enrich = cosine_similarity(df_array, df_array)
    title = indices.index[random.randint(cosine_sim_enrich.shape[0])]

    # Convert the string movie title to a numeric index for our
    # similarity matrix
    b_idx = indices[title]
    # Extract all similarity values computed with the reference movie title
    sim_scores = list(enumerate(cosine_sim_enrich[b_idx]))
    # Sort the values, keeping a copy of the original index of each value
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Select the top-N values for recommendation
    sim_scores = sim_scores[:top_n]
    # Collect indexes
    movie_indices = [i[0] for i in sim_scores]
    # Convert the indexes back into titles
    return titles.iloc[movie_indices]
