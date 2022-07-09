import random
import numpy as np
import pandas as pd
import torch
import pdb
from tqdm import tqdm
from evaluate import MAP, RMSE
import collections
from utils import getNDCG, load_dataset, leave_one_dataset, hitRatioAt10


class film_based_recommender:
    def __init__(self, data_path):
        self.dataset = load_dataset(data_path)

    def calc_sim(self, movie1, movie2):
        genre_vec1 = movie1['genre']
        genre_vec2 = movie2['genre']
        base_sim = np.dot(genre_vec1, genre_vec2)
        if movie1['year'] == movie2['year']:
            base_sim += 0.05
        if movie1['director'] == movie2['director']:
            base_sim += 0.05
        if movie1['writer'] == movie2['writer']:
            base_sim += 0.05
        if len(movie1['stars']) > 0:
            for star in movie1['stars']:
                if star in movie2['stars']:
                    base_sim += 0.05
                    break
        return base_sim

    def read_user(self, ratings, user_ids, m=10):
        movie_records = collections.defaultdict()
        for user in user_ids:
            user_record = ratings[ratings.userId == user]
            user_record = user_record.sort_values(by=['rating', 'timestamp'], ascending=True)
            _m = min(m, len(user_record))
            # movie_record = np.array(user_record.iloc[:_m, 1])
            movie_record = np.array(user_record.iloc[:, 1])
            movie_records[user] = movie_record

        return movie_records

    def read_movies(self):
        movieId_to_imdbId = {}
        imdbId_to_movieId = {}
        for i in range(len(self.dataset[2])):
            movieId, imdbId, _ = self.dataset[2].iloc[i]
            movieId_to_imdbId[movieId] = imdbId
            imdbId_to_movieId[imdbId] = movieId
        genre_record = self.dataset[3].genres.unique()
        genre_types = set()
        # pdb.set_trace()
        for genre in genre_record:
            genre_types = genre_types | set(genre.split('|'))
        genre_types = list(genre_types)
        movie_genre = {}
        for i in range(len(self.dataset[3])):
            movieId, _, genre = self.dataset[3].iloc[i]
            movie_genre[movieId] = genre.split('|')
        movies = {}
        print('construct movie data')
        for i in tqdm(range(len(self.dataset[4]))):
            imdbId, year, director, writer, stars = self.dataset[4].iloc[i, [1, 4, 11, 12, 13]]
            movieId = imdbId_to_movieId[imdbId]
            movies[movieId] = {}
            movies[movieId]['genre'] = np.zeros(len(genre_types))
            for i in range(len(genre_types)):
                if genre_types[i] in movie_genre[movieId]:
                    movies[movieId]['genre'][i] = 1
            movies[movieId]['genre'] /= np.linalg.norm(movies[movieId]['genre'])
            movies[movieId]['year'] = year
            movies[movieId]['director'] = director
            movies[movieId]['writer'] = writer
            if isinstance(stars, str):
                movies[movieId]['stars'] = set(stars.split('|'))
            else:
                movies[movieId]['stars'] = set()

        return movies, movieId_to_imdbId, imdbId_to_movieId, genre_types

    def recommend(self, user_movie_list, ratings):
        user_movies = self.read_user(ratings, list(user_movie_list.keys()))
        movies, movieId_to_imdbId, imdbId_to_movieId, genre_types = self.read_movies()
        # (movieId, sim_score)
        average_sims = collections.defaultdict(dict)
        recommend_list = collections.defaultdict(list)
        print("recommending")
        for user in tqdm(user_movie_list.keys()):
            watched_user_movie = list(ratings[ratings.userId==user].movieId)
            for movie in user_movie_list[user]:
                # if movie in watched_user_movie:
                #     continue
                sims = [self.calc_sim(movies[movie], movies[user_movie]) for user_movie in user_movies[user]]
                average_sims[user][movie] = np.mean(sims)
            recommend_list[user] = list(average_sims[user].keys())
            recommend_list[user].sort(key=lambda x: average_sims[user][x], reverse=True)
        return recommend_list, average_sims


# 直接调用recommend函数
# 输入: 用户id list
# 输出: dict: {user_id: {movie_id1: similarity, movie_id2: ..}, ...}
# example
# users = [1, 2, 3, 4, 5]
# pred = recommend(users)
recommender = film_based_recommender('../data2')
data = leave_one_dataset('../data2')
train_ratings, test_ratings = data[:2]
full_ratings = data[-1]

hit10 = []
hit5 = []
ndcg10 = []
ndcg5 =[]
all_movieIds = np.unique(full_ratings.movieId)
user_movie = collections.defaultdict(list)
for (u, i) in set(zip(test_ratings.userId, test_ratings.movieId)):
    watched_movies = train_ratings.movieId[train_ratings.userId==u]
    user_movie[u].append(i)
    for _ in range(99):
        movie = np.random.choice(all_movieIds)
        while movie in user_movie[u] or movie in watched_movies:
            movie = np.random.choice(all_movieIds)
        user_movie[u].append(movie)
_, average_sim = recommender.recommend(user_movie, train_ratings)
for user in user_movie.keys():
    ratings = []
    for movie in user_movie[user]:
        ratings.append(average_sim[user][movie])
    if 0 in torch.tensor(ratings).topk(10).indices:
        hit10.append(1)
    else:
        hit10.append(0)
    if 0 in torch.tensor(ratings).topk(5).indices:
        hit5.append(1)
    else:
        hit5.append(0)
    rel = np.ones(10)*0
    top10 = torch.tensor(ratings).topk(10).indices
    ndcg10.append(getNDCG(top10.cpu().numpy(), rel))
    rel = np.ones(5)*0
    top5 = torch.tensor(ratings).topk(5).indices
    ndcg5.append(getNDCG(top5.cpu().numpy(), rel))
print("hr10", np.mean(hit10))
print("hr5", np.mean(hit5))
print("ndcg10", np.mean(ndcg10))
print("ndcg5", np.mean(ndcg5))