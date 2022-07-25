# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 18:07:55 2022

@author: misbah.iqbal
"""

import uvicorn
from fastapi import FastAPI
from movie_recommendation import get_movie_recommendations_by_user, get_movie_recommendations_by_genre

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/users/{user_id}")
async def get_recommended_movies_by_user(user_id: int, is_train: bool):
    user_id = int(user_id)
    recommended_movies = get_movie_recommendations_by_user(user_id, is_train)
    print(recommended_movies)
    return recommended_movies.to_dict()

@app.get("/movies/{movie_title}")
async def get_recommended_movies_by_genre(movie_title: str, is_train: bool):
    recommended_movies = get_movie_recommendations_by_genre(movie_title, is_train)
    print(recommended_movies)
    return recommended_movies.to_dict()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)