import random
import time

import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import collections
from utils import getNDCG, load_dataset, hitRatioAt10, leave_one_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from user_based_recommentation import recommend_user
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
img_to_tensor = transforms.ToTensor()
np.random.seed(2022)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)


class MoviesDataset(Dataset):
    def __init__(self, users, items, labels, item_type):
        super().__init__()
        self.users = users
        self.items = items
        self.labels = labels
        self.item_type = item_type

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx], self.item_type[idx]


class NCF(nn.Module):
    def __init__(self, user_num, item_num, extra_feature_len, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, extra_feature_len)
        self.item_embedding = nn.Embedding(item_num, embedding_dim)
        self.user_embedding2 = nn.Linear(extra_feature_len*2, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(in_features=embedding_dim*2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )
        self.output = nn.Sigmoid()

    def forward(self, user_input, item_input, item_types):
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        user_input2 = torch.cat([user_embedding, item_types], dim=-1)
        user_embedding2 = self.user_embedding2(user_input2)
        vector = torch.cat([user_embedding2, item_embedding], dim=-1)
        return self.output(self.net(vector))


class ncf_recommender:
    def __init__(self, user_rec, negative_sample_ratio=4, embedding_dim=8, lr=1e-3):
        self.user_rec = user_rec
        self.negative_sample_ratio = negative_sample_ratio
        self.load_data('../data2')
        self.net = NCF(len(np.unique(self.users)), len(np.unique(self.items)), 20, embedding_dim).to(device)
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # self.model = models.vgg16(pretrained=True).features.to(device)
        # self.model.eval()

    # def extract_feature(self, movieIds, model):
    #     imgs = []
    #     for movieId in movieIds:
    #         img = Image.open(f'./Movie_Poster/{movieId}.jpg')
    #         img = img.resize((56, 56))
    #         imgs.append(img_to_tensor(img).to(device).expand(1, 3, 56, 56))
    #     feature = model(torch.cat(imgs, dim=0))
    #     # pdb.set_trace()
    #     torch.cuda.empty_cache()
    #     return feature.view(feature.shape[0], -1)

    def load_data(self, data_path):
        dataset = leave_one_dataset(data_path)
        train_ratings, test_ratings = dataset[:2]
        self.test_set = set(zip(test_ratings.userId, test_ratings.movieId))
        full_rating = dataset[-1]
        self.all_movie_Ids = np.unique(full_rating.movieId)
        all_user_Ids = np.unique(full_rating.userId)
        self.user2userid = {}
        self.userid2user = {}
        self.movie2movieid = {}
        self.movieid2movie = {}
        for i in range(len(all_user_Ids)):
            self.user2userid[i] = all_user_Ids[i]
            self.userid2user[all_user_Ids[i]] = i
        for i in range(len(self.all_movie_Ids)):
            self.movieid2movie[self.all_movie_Ids[i]] = i
            self.movie2movieid[i] = self.all_movie_Ids[i]

        self.users = []
        self.items = []
        self.labels = []
        self.item_types = []
        genre_record = dataset[3].genres.unique()
        genre_types = set()
        for genre in genre_record:
            genre_types = genre_types | set(genre.split('|'))
        genre_types = list(genre_types)
        self.genre_types = genre_types
        genres = dataset[3]
        self.movie_genre = {}
        for i in range(len(genres)):
            movieId, _, genre = genres.iloc[i]
            self.movie_genre[movieId] = genre.split('|')
        self.user_item_set = set(zip(train_ratings.userId, train_ratings.movieId))
        print("Loading data...")
        for (u, i) in tqdm(self.user_item_set):
            self.users.append(self.userid2user[u])
            self.items.append(self.movieid2movie[i])
            self.labels.append(1)
            item_type = []
            for type in genre_types:
                if type in self.movie_genre[i]:
                    item_type.append(1)
                else:
                    item_type.append(0)
            self.item_types.append(torch.tensor(item_type))
            for _ in range(self.negative_sample_ratio):
                negative_sample = np.random.choice(self.all_movie_Ids)
                while (u, negative_sample) in self.user_item_set:
                    negative_sample = np.random.choice(self.all_movie_Ids)
                self.users.append(self.userid2user[u])
                self.items.append(self.movieid2movie[negative_sample])
                self.labels.append(0)
                item_type = []
                for type in genre_types:
                    if type in self.movie_genre[negative_sample]:
                        item_type.append(1)
                    else:
                        item_type.append(0)
                self.item_types.append(torch.tensor(item_type))

    def train(self, epochs=10):
        model = models.vgg16(pretrained=True).features.to(device)
        model.eval()
        best_HR = np.zeros(10)
        best_HR_merge = np.zeros(10)
        best_NDCG = np.zeros(10)
        best_NDCG_merge = np.zeros(10)
        train_dataset = MoviesDataset(self.users, self.items, self.labels, self.item_types)
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            for data in train_loader:
                user_input, item_input, labels, item_types = data
                user_input = user_input.to(device)
                item_input = item_input.to(device)
                labels = labels.to(device).float()
                item_types = item_types.to(device)
                movieIds = [self.movie2movieid[movie] for movie in item_input.cpu().view(-1).tolist()]
                # pdb.set_trace()
                # img_features = self.extract_feature(movieIds, model)
                # pdb.set_trace()
                pred_prob = self.net(user_input, item_input, item_types)
                loss = self.loss(pred_prob, labels.view(-1, 1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('Evaluating...')
            HR_merge, NDCG_merge = self.evaluate(model)
            for i in range(10):
                # if HR[i] > best_HR[i]:
                #     best_HR[i] = HR[i]
                if HR_merge[i] > best_HR_merge[i]:
                    best_HR_merge[i] = HR_merge[i]
                # if NDCG[i] > best_NDCG[i]:
                #     best_NDCG[i] = NDCG[i]
                if NDCG_merge[i] > best_NDCG_merge[i]:
                    best_NDCG_merge[i] = NDCG_merge[i]
            # print("HR", best_HR)
            print("HR_merge", best_HR_merge)
            # print("NDCG", best_NDCG)
            print("NDCG_merge", best_NDCG_merge)
        return best_HR_merge, best_NDCG_merge

    def evaluate(self, model):
        model.eval()
        HR = []
        HR_merge = []
        NDCG = [] 
        NDCG_merge = []
        hit = [[] for i in range(10)]
        hit_merge = [[] for i in range(10)]
        ndcg = [[] for i in range(10)]
        ndcg_merge = [[] for i in range(10)]
        merge_ratio = 0.5
        for u, i in self.test_set:
            users = [self.userid2user[u]]
            items = [self.movieid2movie[i]]
            item_types = []
            item_type = []
            for type in self.genre_types:
                if type in self.movie_genre[i]:
                    item_type.append(1)
                else:
                    item_type.append(0)
            item_types.append(item_type)
            movieids = [i]
            for _ in range(99):
                random_item = np.random.choice(self.all_movie_Ids)
                while (u, random_item) in self.user_item_set or self.movieid2movie[random_item] in items:
                    random_item = np.random.choice(self.all_movie_Ids)
                users.append(self.userid2user[u])
                items.append(self.movieid2movie[random_item])
                movieids.append(random_item)
                item_type = []
                for type in self.genre_types:
                    if type in self.movie_genre[random_item]:
                        item_type.append(1)
                    else:
                        item_type.append(0)
                item_types.append(item_type)
            users = torch.Tensor(users).long().to(device)
            items = torch.Tensor(items).long().to(device)
            item_types = torch.Tensor(item_types).float().to(device)
            movieIds = [self.movie2movieid[movie] for movie in items.cpu().view(-1).tolist()]
            # img_features = self.extract_feature(movieIds, model)
            pred_prob = self.net(users, items, item_types).cpu().view(-1)

            rating_user = []
            rank = self.user_rec.recommend(u, movieids)
            for movie in movieids:
                rating_user.append(rank[self.user_rec.movieId2movie[movie]])
            rating_user = torch.tensor(rating_user)
            rating_user /= torch.max(rating_user)

            merge_prob = merge_ratio * pred_prob + (1-merge_ratio) * rating_user
            
            for i in range(10):
                # if 0 in pred_prob.view(-1).topk(i+1).indices:
                #     hit[i].append(1)
                # else:
                #     hit[i].append(0)

                if 0 in merge_prob.view(-1).topk(i+1).indices:
                    hit_merge[i].append(1)
                else:
                    hit_merge[i].append(0)
                
                # rel = np.ones(i+1)*0
                # topi = pred_prob.view(-1).topk(i+1).indices
                # ndcg[i].append(getNDCG(topi.cpu().numpy(), rel))
                
                rel = np.ones(i+1)*0
                topi = merge_prob.view(-1).topk(i+1).indices
                ndcg_merge[i].append(getNDCG(topi.cpu().numpy(), rel))
        for i in range(10):
            # HR.append(np.mean(hit[i]))
            HR_merge.append(np.mean(hit_merge[i]))
            # NDCG.append(np.mean(ndcg[i]))
            NDCG_merge.append(np.mean(ndcg_merge[i]))

        return HR_merge, NDCG_merge

def main(embedding_dim=16):
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

    recommmender = ncf_recommender(rec1, negative_sample_ratio=8, embedding_dim=embedding_dim, lr=1e-4)
    HR, NDCG = recommmender.train(epochs=30)
    return HR, NDCG


if __name__ == '__main__':
    HR = []
    NDCG = []
    Embedding = [4, 8, 12, 20]
    for embedding in Embedding:
        _HR, _NDCG = main(embedding_dim = embedding)
        HR.append(_HR)
        NDCG.append(_NDCG)
    for i in range(4):
        print("Embedding", Embedding[i])
        print("HR")
        print(HR[i])
        print("NDCG")
        print(NDCG[i])
    

