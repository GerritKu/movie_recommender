import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import random
from model_fit.utils import movie_path as movie_path
from model_fit.utils import rating_path as rating_path 
from model_fit.utils import my_ratings as rating_dict
from model_fit.utils import (create_initial_matrix, get_similarity, get_top_simils, 
                             get_unseen, get_recommendation,recommend_nmf)

nav = st.sidebar.radio(
    "Welcome",
    ["home", "your data", "get recommendations"]
    ) 

#initialise df for data adaption
df = pd.DataFrame.from_dict(rating_dict, orient="index", columns=["rating"])
df.index.name='movie'

if nav == "home":
    
   # Create Header Image and title
   col1, col2, col3 = st.columns(3)

   with col1:
      st.write(' ')

   with col2:
      st.title("Movie Recommender")
      st.image("https://images.unsplash.com/photo-1627133805103-ce2d34ccdd37?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80", width=400)
      st.write(" ")
      st.write("Recommendations can be created via your data (adapt on left) or be based on the following default data.")
      
   st.dataframe(df, use_container_width=True)

   with col3:
      st.write(' ')


if nav == "your data":
   st.write('Please provide some ratings from 0 (bad) to 5 (very good) as input for your recommendations')

   with st.form("my_form"):
      movies = pd.read_csv(movie_path)
      titles_list = list(movies['title'])
      cola, colb = st.columns(2)
      with cola:
         movie1 = st.selectbox('Pick a Movie title',(titles_list), key = "movie1")
      with colb:
         rating1 = st.radio('your rating',(range(0,6)),horizontal = True, key = "rating1")
      colc, cold = st.columns(2)
      with colc:
         movie2 = st.selectbox('Pick a Movie title',(titles_list), key = "movie2")
      with cold:
         rating2 = st.radio('your rating',(range(0,6)),horizontal = True, key = "rating2")
      cole, colf = st.columns(2)
      with cole:
         movie3 = st.selectbox('Pick a Movie title',(titles_list), key = "movie3")
      with colf:
         rating3 = st.radio('your rating',(range(0,6)),horizontal = True, key = "rating3")

      # Every form must have a submit button.
      submitted = st.form_submit_button("Save")
      if submitted:
         data_dict = {movie1 : rating1,
                       movie2 : rating2,
                       movie3 : rating3
                       }
         st.session_state['user_data'] = data_dict
         st.write("data saved. Proceed to recommendations")
         
if nav == "get recommendations":
   if 'user_data' not in st.session_state:
      st.session_state['user_data'] = rating_dict
   col1, col2, col3 = st.columns(3)
   with col1:
      st.write(' ')
   with col2:
      recommender = st.radio(
      "Please choose", ('Cosine similarity', 'NMF')) 
      
      if st.button('Get recommendations'):
         # Pre-Load variables and recommendations
         MODEL_FILE = "./artefacts/nmf_29_42_1111"

         if recommender == 'Cosine similarity':
            st.write('Creating recommendations based on Cosine similarity')
            #Create initial matrix for cosine similarity
            df_cosine = create_initial_matrix(movie_path,rating_path,normalize=True)
            #get recommendations
            recommendation_cosine = get_recommendation(st.session_state['user_data'],df_cosine,k=100,n=5)
            st.write(f'#1 {recommendation_cosine[0]}')
            st.write(f'#2 {recommendation_cosine[1]}')
            st.write(f'#3 {recommendation_cosine[2]}')

         elif recommender == 'NMF':
            st.write('Creating recommendations based on NMF')
            #get recommendations
            recommendation_nmf = recommend_nmf(st.session_state['user_data'],MODEL_FILE,n=3)
            st.write(f'#1 {recommendation_nmf[0]}')
            st.write(f'#2 {recommendation_nmf[1]}')
            st.write(f'#3 {recommendation_nmf[2]}')

         else: 
            st.write('Please make a selection.')

   with col3:
         st.write(' ')
