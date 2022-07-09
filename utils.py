import pdb

import numpy as np
import pandas as pd
import torch


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


# rank_list: rank result
# pos_items: items that should be recommended
def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    rel = np.zeros_like(pos_items)
    rel[0] = 1
    idcg = getDCG(rel)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


# test_list: {user_id1: list of ordered recommends, user_id2...}
# recommend_list: the same
# def getMeanNDCG(test_list, recommend_list):
#     ndcg_tot = []
#     for user in test_list.keys():
#         ndcg_tot.append(getNDCG(test_list[user], recommend_list[user]))

#     return np.mean(ndcg_tot)


# nearest_item: {user_id1: movieId, user_id2...}
# recommend_rating: pred_rating_mat where item in training is zero
def hitRatioAt10(nearest_item, recommend_rating, userId2user, user2userId, movieId2movie, movie2movieId):
    hit = []
    rating_loss = 0
    recommend_rating = recommend_rating.cpu().numpy()
    for userid in nearest_item.keys():
        user = userId2user[userid]
        non_train_item = np.argwhere(recommend_rating[user] > 0).reshape(-1)
        random_99_item = non_train_item[np.random.randint(0, len(non_train_item), 99)].tolist()
        movieId, rating = nearest_item[userid]
        movie = movieId2movie[movieId]
        test_item_list = random_99_item + [movie]
        test_item_rating = torch.from_numpy(recommend_rating[user][test_item_list])
        if 99 in test_item_rating.topk(10).indices:
            hit.append(1)
        else:
            hit.append(0)
        rating_loss += (recommend_rating[user][movie] - rating)**2
    return np.mean(hit), rating_loss


def load_dataset(data_path):
    ratings = pd.read_csv(data_path + '/ratings.csv')  # 用户对电影评分
    links = pd.read_csv(data_path + '/links.csv')  # 电影在网站上链接号
    genres = pd.read_csv(data_path + '/movies.csv')  # 电影类型
    attributes = pd.read_csv(data_path + '/crawl_data.csv')  # 电影特征
    users = np.unique(ratings.iloc[:, 0])
    train_set = []
    test_set = []
    for user in users:
        user_ratings = ratings[ratings.userId == user]
        if len(user_ratings) > 100:
            recent_watches = user_ratings.sort_values(by=['timestamp'], ascending=False)
            test_set += list(recent_watches.iloc[:50].index)
            train_set += list(recent_watches.iloc[50:].index)
        else:
            train_set += list(user_ratings.index)
    test_set = np.array(test_set)[np.array(ratings.iloc[test_set].rating) > 3].tolist()


    train_ratings = ratings.iloc[train_set, :]
    test_ratings = ratings.iloc[test_set, :]

    return train_ratings, test_ratings, links, genres, attributes, ratings


def leave_one_dataset(data_path):
    ratings = pd.read_csv(data_path + '/ratings.csv')  # 用户对电影评分
    links = pd.read_csv(data_path + '/links.csv')  # 电影在网站上链接号
    genres = pd.read_csv(data_path + '/movies.csv')  # 电影类型
    attributes = pd.read_csv(data_path + '/crawl_data.csv')  # 电影特征
    users = np.unique(ratings.iloc[:, 0])
    train_set = []
    test_set = []
    for user in users:
        user_ratings = ratings[ratings.userId == user]
        recent_watches = user_ratings.sort_values(by=['timestamp'], ascending=False)
        # pdb.set_trace()
        test_set.append(recent_watches.index[0])
        train_set += list(recent_watches.iloc[1:].index)
    # pdb.set_trace()
    train_ratings = ratings.iloc[train_set, :]
    test_ratings = ratings.iloc[test_set, :]

    return train_ratings, test_ratings, links, genres, attributes, ratings