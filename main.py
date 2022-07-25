# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 18:07:55 2022

@author: misbah.iqbal
"""

import uvicorn
from fastapi import FastAPI, Form,Request
from starlette.responses import FileResponse
from movie_recommendation import get_movie_recommendations_by_user, get_movie_recommendations_by_genre
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/users")
async def get_user_movies(request: Request):
    return templates.TemplateResponse("recommendation_by_id.html",{"request": request})


@app.get("/movies")
async def get_movie_neighbours(request: Request):
    return templates.TemplateResponse("movie_neighbour.html",{"request": request})


@app.post("/movies-neighbour")
async def get_recommended_movies_by_genre(request: Request, movie_title: str= Form(), is_train: bool= Form()):
    recommended_movies = get_movie_recommendations_by_genre(movie_title, is_train)
    print(recommended_movies)
    recommended_movies = recommended_movies.transpose()
    return templates.TemplateResponse("user_movie.html",{"request": request,"recommended_movies":recommended_movies})


@app.post("/user-fav")
async def get_recommended_movies_by_user(request: Request, user_id: int = Form(), is_train:bool = Form()):
    print(is_train,user_id)
    recommended_movies = get_movie_recommendations_by_user(user_id, is_train)
    recommended_movies = recommended_movies.transpose()
    return templates.TemplateResponse("user_movie.html",{"request": request,"recommended_movies":recommended_movies})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)