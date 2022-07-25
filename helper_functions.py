import tensorflow as tf
from CF_model import CFModel
import numpy as np
import pandas as pd
import pickle

def mask(df, key, function):
    """Returns a filtered dataframe, by applying function to key"""
    return df[function(df[key])]

def flatten_cols(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df


def movie_genre(movies,genre_cols):
    def get_all_genres(gs):
        active = [genre_cols for genre_cols, g in zip(genre_cols, gs) if g==1]
        if len(active) == 0:
            return 'Other'
        return '-'.join(active)
    movies['all_genres'] = [get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genre_cols])]

def split_dataframe(df, holdout_fraction=0.1):
  test = df.sample(frac=holdout_fraction, replace=False)
  train = df[~df.index.isin(test.index)]
  return train, test

def build_rating_sparse_tensor(ratings_df, users_rows, movies_rows):

  indices = ratings_df[['user_id', 'movie_id']].values
  # print(indices)
  values = ratings_df['rating'].values
  result = tf.SparseTensor(
      indices=indices,
      values=values,
      dense_shape=[users_rows, movies_rows])
  # print(result)
  return result

def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
  # print(sparse_ratings)
  # print(user_embeddings)
  # print(movie_embeddings)
  predictions = tf.gather_nd(
      tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
      sparse_ratings.indices)
  loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
  return loss

def build_model(ratings, embedding_dim, init_stddev, users_rows, movies_rows):
  train_ratings, test_ratings = split_dataframe(ratings)
  # print(train_ratings)
  A_train = build_rating_sparse_tensor(train_ratings, users_rows, movies_rows)
  A_test = build_rating_sparse_tensor(test_ratings, users_rows, movies_rows)
  user_embedding =tf.random.normal([A_train.dense_shape[0], embedding_dim],stddev = init_stddev)
  movie_embedding = tf.Variable(tf.random.normal([A_train.dense_shape[1], embedding_dim],stddev = init_stddev))
  train_loss = sparse_mean_square_error(A_train, user_embedding, movie_embedding)
  test_loss = sparse_mean_square_error(A_test, user_embedding, movie_embedding)
  metrics = {
      'train_error': train_loss,
      'test_error': test_loss
  }

  embeddings = {
      'user_id' : user_embedding,
      'movie_id': movie_embedding
  }
  new1 = CFModel(embeddings, train_loss, [metrics])
  # print(new1.embeddings)

  return new1


def compute_scores(query_embedding, item_embeddings, measure):

  u = query_embedding
  V = item_embeddings
  if measure == 'cosine':
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    u = u / np.linalg.norm(u)
  scores = u.dot(V.T)
  return scores

def user_recommendations(embeddings, user_id, movies, ratings, measure='cosine', exclude_rated=False, k=6):
  USER_RATINGS = True
  if USER_RATINGS:
    scores = compute_scores(
        embeddings["user_id"][user_id], embeddings["movie_id"], measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
        score_key: list(scores),
        'movie_id': movies['movie_id'],
        'titles': movies['title'],
        'genres': movies['all_genres'],
    })
    if exclude_rated:
      # remove movies that are already rated
      rated_movies = ratings[ratings.user_id == "943"]["movie_id"].values
      df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]
    df.sort_values(by=score_key, ascending=False)
    user_rating_data = df.sort_values(by=score_key, ascending=False)
    return user_rating_data[:k]

def movie_neighbors(embeddings, title_substring, movies, measure='cosine', k=6):
  # Search for movie ids that match the given substring.
  ids =  movies[movies['title'].str.contains(title_substring)].index.values
  titles = movies.iloc[ids]['title'].values
  if len(titles) == 0:
    raise ValueError("Found no movies with title %s" % title_substring)
  print("Nearest neighbors of : %s." % titles[0])
  if len(titles) > 1:
    print("[Found more than one matching movie. Other candidates: {}]".format(
        ", ".join(titles[1:])))
  movie_id = ids[0]
  scores = compute_scores(
      embeddings["movie_id"][movie_id], embeddings["movie_id"],
      measure)
  score_key = measure + ' score'
  df = pd.DataFrame({
      score_key: list(scores),
      'titles': movies['title'],
      'genres': movies['all_genres']
  })
  recommended_movies = df.sort_values(by=score_key, ascending=False)
  return recommended_movies[:k]


def save_model(file_path, model):
  with open(file_path, 'wb') as files:
    pickle.dump(model,files, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_model(file_path):
  with open(file_path, 'rb') as files:
    model = pickle.load(files)
    return model