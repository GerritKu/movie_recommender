import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import random

from sklearn.decomposition import NMF 
from sklearn.metrics.pairwise import cosine_similarity

################################################################################################

#Variables
PATH = Path(__file__).parent/'data'
movie_path = Path(f'{PATH}/movies.csv')
rating_path = Path(f'{PATH}/ratings.csv')

my_ratings = {
    'Iron Man 2 (2010)' : 4.5,
    '22 Jump Street (2014)' : 4,
    'Over the Hedge (2006)' : 4,
    'Mission: Impossible (1996)' :3.4,
    'Frozen (2013)' : 4,
    'Saw (2004)' : 4.5,
    'Jumanji: Welcome to the Jungle (2017)' : 0.5,
    'Asterix & Obelix vs. Caesar (Astérix et Obélix contre César) (1999)' : 0.5,
    'Forrest Gump (1994)' : 3.5,
    'Gladiator (2000)' : 2.5,
    'Lord of the Rings: The Fellowship of the Ring, The (2001)': 4,
    'Dark Knight, The (2008)' : 5,
    'Lord of the Rings: The Return of the King, The (2003)' : 3.7,
    'Inception (2010)' : 5,
    'Shutter Island (2010)' : 4.5,
    'City of God (Cidade de Deus) (2002)':3,
    'Lord of the Rings: The Two Towers, The (2002)': 4,
    'Dark Knight Rises, The (2012)':5,
    'Braveheart (1995)': 4.5,
    'Inglourious Basterds (2009)' : 4.5
}

################################################################################################

#Functions

def recommend_nmf(query, model, n=10):
    
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.

    Args:
        query (dict): The ratings of a new user with keys = title = str and values = rating = int
        model (str): Path to pretrained nmf model to be loaded via pickle
        n (optional,int): Number of recommendations returned

    Returns:
        list: list of n titles for most recommended movies.
    """
    with open(f'{model}.pkl','rb') as file:
        loaded_model = pickle.load(file)

    recommendations = []
    # 1. candidate generation
    movies = loaded_model.feature_names_in_
    # 2. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=movies, index=["new_user"])
    new_user_dataframe_imputed = new_user_dataframe.fillna(3)

    # 3. scoring
    p_new_user_matrix = loaded_model.transform(new_user_dataframe_imputed)
    
    # calculate the score with the NMF model
    r_hat_new_user_matrix = np.dot(p_new_user_matrix,loaded_model.components_)
    
    # 4. ranking
    r_hat_new_user = pd.DataFrame(data=r_hat_new_user_matrix,
                         columns=movies,
                         index = ['new_user'])
    sorted_list = r_hat_new_user.transpose().sort_values(by="new_user", ascending=False).index.to_list()
    
    # filter out movies already seen by the user
    rated_movies = list(query.keys())
    recommended = [movie for movie in sorted_list if movie not in rated_movies]
    
    # return the top-k highest rated movie ids or titles
    recommendations = recommended[0:n]
    
    return recommendations

################################################################################################################
################################################################################################################

def create_initial_matrix(movie_path,rating_path,normalize=True):
    """This function loads a movie and rating .csv from the movie lens data set 
    to create a cleaned movies x users matrix with respective ratings

    Args:
        movie_path (string): path to movie.csv
        rating_path (string): path to ratings.csv

    Returns:
        DataFrame: cleaned movies x users matrix with respective ratings
    """
    #check if input paths exist
    assert os.path.exists(movie_path)
    assert os.path.exists(rating_path)

    #read in data for movies and ratings based on movie_lense data set
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)

    #remove duplicates
    movies["title"].drop_duplicates(inplace = True)

    #merge data to an initial movies x users dataframe
    initial = ratings.merge(right=movies,how = 'inner', on="movieId")
    initial.sort_values("userId",inplace=True)
    initial.set_index("userId",inplace=True)
    initial.drop(["movieId","timestamp","genres"],axis = 1, inplace=True)
    initial = pd.pivot_table(initial, index = "userId", columns = "title", values = "rating").T

    if normalize:
        average = initial.copy()
        for column in initial.columns:
            average[column] = initial[column] - initial[column].mean()
        initial = average

    return initial

################################################################################################################
################################################################################################################

def get_similarity(rating_dict,df):
    """this function creates the cosine_similarity between a new user and the users 
    in the initial data based on user ratings.

    Args:
        rating_dict (dictionary): The ratings of a new user with keys = title = str and values = rating = int
        df (Dataframe): initial matrix of movies x users and their ratings from movie lens data

    Returns:
        Dataframe: cosine_similarity between a new user and the users 
    in the initial data (users x users)
    """
    
    #create matrix for new user ratings
    new_user_item = pd.DataFrame(rating_dict, index = [df.columns[-1]+1]).T

    #combine initial matrix with the new user ratings
    user_item = pd.concat([df,new_user_item],axis=1)
    
    #Get rid of NaN with constant = 0
    user_item.fillna(0,inplace=True)

    #create cosine_similarity matrix
    user_user = cosine_similarity(user_item.T)
    user_user = pd.DataFrame(user_user, columns = user_item.columns, index = user_item.columns)

    return user_user

################################################################################################################
################################################################################################################

def get_top_simils(user_user,k=10):
    """This function returns the top k most simialar users based on cosine similarity of movie ratings.

    Args:
        user_user (Dataframe): Matrix with cosine similarity of users (users x users)
        k (int, optional): Number for the length of list of simialar users. Defaults to 10.

    Returns:
        pd.Series: Series/List of most simialar users based on on cosine similarity of movie ratings.
    """
    top_k_users = user_user[user_user.columns[-1]].sort_values(ascending=False).index[1:k+1]
    return top_k_users

################################################################################################################
################################################################################################################

def get_unseen(rating_dict,df):
    """Get a set of all movies the user did not rate (Thesis: not watch) yet.

    Args:
        rating_dict (dictionary): The ratings of a new user with keys = title = str and values = rating = int
        df (Dataframe): initial matrix of movies x users and their ratings from movie lens data

    Returns:
        Set: Set of all movies in the database, the new user has not rated / seen yet.
    """
    seen_set = set(list(rating_dict.keys()))
    movies_set = set(list(df.index))
    unseen = seen_set.symmetric_difference(movies_set)

    return unseen

################################################################################################################
################################################################################################################

def get_recommendation(rating_dict,df,k=10,n=5):
    """This function recommends n movies based on a new user rating being compared to the top k users 
    with highest cosine similarity scores.

    Args:
        rating_dict (dictionary): The ratings of a new user with keys = title = str and values = rating = int
        df (Dataframe): initial matrix of movies x users and their ratings from movie lens data
        k (int, optional): Number for the length of list of simialar users. Defaults to 10.
        n (int, optional): Number of prioritized recommendations. Defaults to 5.

    Returns:
        pd.series: Series of n titles for most recommended movies.
    """
    #Get rid of NaN with constant = 0
    user_item = df.fillna(0)
    #create set of unseen movies
    unseen = get_unseen(rating_dict,df)
    #get cosine similarity
    user_user = get_similarity(rating_dict,df)
    # get users with highest similarity
    top_simils = set(get_top_simils(user_user,k))
    #init an empty dict to fill with recommendations
    recommendation_dic = {}
    #get all users, which rated unseen movies
    for movie in list(unseen):
        other_users = df.columns[~df.loc[movie].isna()]
        other_users = set(other_users)
        #initialise numerator and denominator
        num = 0
        den = 0
        #get ratings for the unseen movie only of users with highest cosine similarity
        if len(list(other_users.intersection(top_simils))) != 0:
            for other_user in other_users.intersection(top_simils):
                rating = user_item[other_user][movie]
                sim = user_user[611][other_user]
                num = num + (rating*sim)
                den = den + sim 
            
            #Create mean ratio for one unseen movie (added constant to prevent division by 0)
            ratio = num/(den + 0.0000000001)

            recommendation_dic[movie] = ratio
    recommendation = pd.DataFrame(recommendation_dic, index = ["rating"]).T.sort_values("rating", ascending = False)
        
    return recommendation.iloc[0:n].index