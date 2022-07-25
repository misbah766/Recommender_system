import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from helper_functions import movie_genre, build_model, compute_scores, user_recommendations,movie_neighbors, save_model, load_model
import collections


tf.disable_v2_behavior()
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.3f}'.format
DOT = 'dot'
COSINE = 'cosine'

 # =============================================================================
# reading data
# =============================================================================
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data_set/u.user',sep='|',names = users_cols, encoding = 'latin-1')


ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data_set/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + genre_cols
movies = pd.read_csv('data_set/u.item',sep = '|',names = movies_cols, encoding = 'latin-1')



# =============================================================================
# preprocessing data
# =============================================================================

users["user_id"] = users["user_id"].apply(lambda x: str(x-1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))
movies["year"] = movies["release_date"].apply(lambda x:str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x-1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x-1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

genre_occurances = movies[genre_cols].sum().to_dict()
movie_genre(movies,genre_cols)


movie_data = ratings.merge(movies,on='movie_id').merge(users,on='user_id')

def get_model(TRAIN=False):
  model = build_model(ratings, embedding_dim=30, init_stddev=0.5, users_rows=users.shape[0], movies_rows=movies.shape[0])
  if TRAIN:
    model.train(num_iterations=1000, learning_rate=10.)
    # scores = compute_scores(model.embeddings["user_id"][43], model.embeddings["movie_id"], measure=COSINE)
    
    save_model('models/model_embeddings.pkl', model.embeddings)
  
  embeddings = load_model('models/model_embeddings.pkl')
  return embeddings
    


def get_movie_recommendations_by_user(user_id, TRAIN):
    model = get_model(TRAIN)
    user_rating_data = user_recommendations(model, user_id, movies, ratings)
    return user_rating_data

def get_movie_recommendations_by_genre(movie_title, TRAIN):
    model = get_model(TRAIN)
    user_rating_data = movie_neighbors(model, movie_title, movies, measure='cosine', k=6)
    return user_rating_data

# =============================================================================
# predict recommendations
# =============================================================================
if __name__ == "__main__":
    embeddings = get_model(TRAIN=False)
    user_rating_data = user_recommendations(embeddings, 44, movies, ratings)
    print(user_rating_data)


    movie_neighbors = movie_neighbors(embeddings, "Mr. Holland's Opus", movies)
    print(movie_neighbors)


