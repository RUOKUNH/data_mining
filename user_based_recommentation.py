import pdb
import re
import numpy as np
import pandas as pd
from operator import itemgetter
import random
import torch
from tqdm import tqdm
from evaluate import MAP, RMSE
import collections
from utils import getNDCG, load_dataset, leave_one_dataset, hitRatioAt10


class recommend_user():
    def __init__(self, data, k_user_num=20, k_movie_num=5):
        self.data = data
        self.k_user_num = k_user_num  # 选择与用户相近的K个用户
        self.k_movie_num = k_movie_num  # 给用户推荐电影数目
        self.movie_2_user = {}  # 电影-用户非稀疏集合
        self.user_sim = {}  # 用户相似度集合
        self.movie_count = 0  # 电影总数
        self.user_count = 0  # 用户总数
        self.user_2_movie = []  # 用户-电影索引矩阵
        self.movie_set = []
        self.user_set = []

    # 将原始数据进行转换成用户-电影矩阵
    def read_data(self, training_data):
        full_ratings = self.data[-1]
        self.user_set = np.unique(full_ratings.userId).tolist()
        self.movie_set = np.unique(full_ratings.movieId).tolist()
        self.movie_count = len(self.movie_set)
        self.user_count = len(self.user_set)
        user_2_movie = np.zeros((self.user_count, self.movie_count))
        for i in range(self.user_count):
            for j in range(self.movie_count):
                if (self.user_set[i], self.movie_set[j]) in training_data:
                    user_2_movie[i][j] = training_data[(self.user_set[i], self.movie_set[j])]
        self.user_2_movie = user_2_movie
        self.movieId2movie = {}
        for i in range(self.movie_count):
            self.movieId2movie[self.movie_set[i]] = i

    # 该函数计算用户的相似度
    def cal_user_sim(self, training_data):
        movie_2_user = {}
        # 根据电影构建索引, 解决原来矩阵稀疏性问题
        for x in training_data:
            user = x[0]
            movie = x[1]
            if movie not in movie_2_user:
                movie_2_user[movie] = set()
            movie_2_user[movie].add(user - 1)  # u-1
        self.movie_2_user = movie_2_user
        # 给每位用户找出至少有一部共同评分电影的用户集合
        self.user_sim = np.zeros((self.user_count, self.user_count))
        u_neighbor = {}
        for movie, users in movie_2_user.items():
            for u in users:
                if u not in u_neighbor:
                    u_neighbor[int(u)] = set()
                for v in users:
                    if int(u) != int(v) and int(v) not in u_neighbor[int(u)]:
                        u_neighbor[int(u)].add(int(v))
        # 计算用户间的相似度
        for u in range(self.user_count):
            for v in u_neighbor[u]:
                self.user_sim[u][v] = self.user_2_movie[u].dot(self.user_2_movie[v].T) / (
                            np.sqrt(self.user_2_movie[u].dot(self.user_2_movie[u].T)) * np.sqrt(
                        self.user_2_movie[v].dot(self.user_2_movie[v].T)))
                self.user_sim[v][u] = self.user_2_movie[u].dot(self.user_2_movie[v].T) / (
                            np.sqrt(self.user_2_movie[u].dot(self.user_2_movie[u].T)) * np.sqrt(
                        self.user_2_movie[v].dot(self.user_2_movie[v].T)))

    # 选择k个近邻对电影评分预测后推荐
    def recommend(self, user, movie_list):
        k = self.k_user_num
        rank = {}
        other = []
        u = user - 1
        for v in range(self.user_count):
            if v != u:
                other.append((self.user_sim[u][v], v))
        v_set = sorted(other, reverse=True)[0:k]
        belief = 0
        for i in range(k):
            belief += v_set[i][0]
        belief /= k
        for movieid in movie_list:
            try:
                movie = self.movieId2movie[movieid]
            except:
                pdb.set_trace()
            num = []
            den = 1e-8
            for x in v_set:
                v = x[1]
                if self.user_2_movie[v][movie] == 0:
                    continue
                num.append(self.user_sim[u][v] * self.user_2_movie[v][movie])
                den += self.user_sim[u][v]
            rank.setdefault(movie, np.sum(num) / den)
        return rank


def evaluate(recmmender):
    data = leave_one_dataset('../data2')
    train_ratings, test_ratings = data[:2]
    full_ratings = data[-1]
    hit10 = []
    hit5 = []
    ndcg10 = []
    ndcg5 =[]
    hit_popular = []
    watch_count = {}
    for movie in full_ratings.movieId:
        watch_count[movie] = len(full_ratings[full_ratings.movieId==movie])
    popular_movie = list(watch_count.keys())
    popular_movie.sort(key=lambda x: watch_count[x], reverse=True)
    all_movieIds = np.unique(full_ratings.movieId)
    user_movie = collections.defaultdict(list)
    for (u, i) in tqdm(set(zip(test_ratings.userId, test_ratings.movieId))):
        watched_movies = train_ratings.movieId[train_ratings.userId == u]
        for _ in range(99):
            movie = np.random.choice(all_movieIds)
            while movie in user_movie[u] or movie in watched_movies:
                movie = np.random.choice(all_movieIds)
            user_movie[u].append(movie)
        user_movie[u].append(i)
        rank = recmmender.recommend(u, user_movie[u])
        ratings = []
        for movie in user_movie[u]:
            ratings.append(rank[recmmender.movieId2movie[movie]])
        if 99 in torch.tensor(ratings).topk(10).indices:
            hit10.append(1)
        else:
            hit10.append(0)
        if 99 in torch.tensor(ratings).topk(5).indices:
            hit5.append(1)
        else:
            hit5.append(0)
        rel = np.ones(10)*99
        top10 = torch.tensor(ratings).topk(10).indices
        ndcg10.append(getNDCG(top10.cpu().numpy(), rel))
        rel = np.ones(5)*99
        top5 = torch.tensor(ratings).topk(5).indices
        ndcg5.append(getNDCG(top5.cpu().numpy(), rel))
    print("hr10", np.mean(hit10))
    print("hr5", np.mean(hit5))
    print("ndcg10", np.mean(ndcg10))
    print("ndcg5", np.mean(ndcg5))

def main():
    target_user = 20
    data = leave_one_dataset('../data2')
    train_ratings = data[0]
    full_rating = data[-1]
    rating_samples = np.array(train_ratings)
    training_data = {}
    for x in rating_samples:  # 将数据格式转换成（（用户，电影）:评分）
        training_data.setdefault((x[0], x[1]), x[2])
    # 基于用户评分协同过滤推荐
    rec1 = recommend_user(data, k_movie_num=5)
    rec1.read_data(training_data)
    rec1.cal_user_sim(training_data)
    evaluate(rec1)


if __name__ == '__main__':
    main()