# movie_recommender using NMF or cosine similarity deployed with streamlit

This repository includes the code required to deploy <a href="https://gerritku-movie-recommender-app-rwya5g.streamlit.app/">this movie recommender site.</a>
The small app provides recommendations based on Non-Negative Matrix factorization (NMF) models or a cosine similarity based on the user's input. Deployed using Streamlit. 

## Structure of the repository

### 1. Installation

Download/Clone and pip install the libraries listed in the [requirements](requirements.txt) file. This code is written in python 3.9

### 2. Code

- [app](app.py) is the main file of streamlit application running here or to be locally deployed. 
- [artefacts](./artefacts/) include the pretrained NMF models on the movielense with different parameter settings. Saved and re-loaded with pickle.
- [utils.py](utils.py) includes static variables like paths and my default movie ratings as well as all functions needed to run the movie recommendations.

### 3. Data
- [data](./data/) includes the movie and ratings data form the movie lens dataset aquired <a href="ahttps://grouplens.org/datasets/movielens/.">here</a>

### 4. Results
This repository demonstrates the use of recommender algorithms to generate movie recommendations based on user's preference deployed in a small web-app.