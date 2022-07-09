import random
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import collections
from utils import getNDCG, load_dataset, hitRatioAt10, leave_one_dataset
import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)


class funkSVD:
    def __init__(self, data_path):
        self.dataset = leave_one_dataset(data_path)
        self.user_number = len(np.unique(self.dataset[-1].userId))
        self.movie_number = len(np.unique(self.dataset[-1].movieId))
        self.rating_mat = torch.zeros(self.user_number, self.movie_number).to(device)
        self.movieId2movie = {}
        self.movie2movieId = {}
        self.userId2user = {}
        self.user2userId = {}
        for i in range(self.user_number):
            self.userId2user[i + 1] = i
            self.user2userId[i] = i + 1
        movies = np.unique(self.dataset[-1].movieId)
        for i in range(len(movies)):
            self.movieId2movie[movies[i]] = i
            self.movie2movieId[i] = movies[i]
        self.train_rating = self.dataset[0]
        for i in range(len(self.train_rating)):
            userId, movieId, rating = self.train_rating.iloc[i, :3]
            user, movie = self.userId2user[userId], self.movieId2movie[movieId]
            self.rating_mat[int(user), int(movie)] = rating
        self.test_rating = self.dataset[1]
        self.nearest_items = {}
        for i in range(len(self.test_rating)):
            userId, movieId, rating = self.test_rating.iloc[i, :3]
            self.nearest_items[userId] = (movieId, rating)
        self.k = 5
        self.user_mat = torch.from_numpy(np.random.random((self.user_number, self.k))).to(device)
        self.movie_mat = torch.from_numpy(np.random.random((self.k, self.movie_number))).to(device)
        self.lr = 1e-4
        self._lambda = 3e-3

    def train(self, epochs=500):
        best_hitRatio = np.zeros(2)
        best_NDCG = np.zeros(2)
        for epoch in range(epochs):
            pred_rating = torch.mm(self.user_mat, self.movie_mat)
            pred_rating[self.rating_mat == 0] = 0
            loss = torch.sum((pred_rating - self.rating_mat) ** 2) + \
                   self._lambda * (torch.sum(self.user_mat ** 2) + torch.sum(self.movie_mat ** 2))
            delta_user_mat = (pred_rating - self.rating_mat).mm(self.movie_mat.T) + self._lambda * self.user_mat
            delta_movie_mat = (pred_rating - self.rating_mat).T.mm(self.user_mat).T + self._lambda * self.movie_mat
            self.user_mat -= self.lr * delta_user_mat
            self.movie_mat -= self.lr * delta_movie_mat
            if (epoch + 1) % 50 == 0:
                HR, NDCG, hr10 = self.valid()
                for i in range(2):
                    if HR[i] > best_hitRatio[i]:
                        best_hitRatio[i] = HR[i]
                    if NDCG[i] > best_NDCG[i]:
                        best_NDCG[i] = NDCG[i]
                print("Epoch", epoch+1)
                print("Best HR", best_hitRatio)
                print("Best NDCG", best_NDCG)
                print(hr10)

    def valid(self):
        HR = []
        NDCG = []
        pred_rating = torch.mm(self.user_mat, self.movie_mat)
        pred_rating[self.rating_mat > 0] = 0
        # pdb.set_trace()
        for i in range(2):
            hit = []
            ndcg = []
            for userid, movieid in set(zip(self.test_rating.userId, self.test_rating.movieId)):
                user = self.userId2user[userid]
                movie = self.movieId2movie[movieid]
                user_rating = pred_rating[user]
                non_train_item = np.argwhere(user_rating.cpu().numpy() > 0).reshape(-1)
                random_99_item = np.random.choice(non_train_item, 99).tolist()
                test_samples = torch.tensor(random_99_item + [movie])
                topi = user_rating[test_samples].topk((i+1)*5).indices
                if 99 in topi:
                    hit.append(1)
                else:
                    hit.append(0)
                rel = np.ones((i+1)*5)*99
                ndcg.append(getNDCG(topi.cpu().numpy(), rel))
            HR.append(np.mean(hit))
            NDCG.append(np.mean(ndcg))
        hitRatio = hitRatioAt10(self.nearest_items, pred_rating, self.userId2user, self.user2userId, self.movieId2movie,
                                self.movie2movieId)
        return HR, NDCG, hitRatio


recommender = funkSVD('../data2')
recommender.train(epochs=10000)
np.savetxt('../checkpoints/user_mat.txt', np.array(recommender.user_mat))
np.savetxt('../checkpoints/movie_mat.txt', np.array(recommender.movie_mat))
# pdb.set_trace()
# recommender.valid()
