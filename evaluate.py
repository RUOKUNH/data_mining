import numpy as np
import sys

# Root Mean Squared Error，RMSE, 只适用于评分预测
# pred in form {user_id: {movie_id1: rating, movie_id2: rating, ...}, ...}
# dataset in form the same as pred
def RMSE(user_rating_pred, test_dataset):
    RSEs = []
    for usr in test_dataset.keys():
        if usr not in user_rating_pred.keys():
            raise KeyError(f'user {usr} has not been predicted')
        square_err = 0
        for movie in test_dataset[usr].keys():
            real_rating = test_dataset[usr][movie]
            if movie not in user_rating_pred[usr].keys():
                raise KeyError('movie {movie} of user {usr} has not been predicted')
            pred_rating = user_rating_pred[usr][movie]
            square_err += (pred_rating - real_rating) ** 2
        square_err /= len(test_dataset[usr])
        RSEs.append(np.sqrt(square_err))
    return np.mean(RSEs)


# Mean Average Precision (MAP)
# user_recommend in form {user_id: {movie_id: recommend order} (所有电影推荐位序), ...}
# test_dataset in form {user_id: recommend_list(of movie_ids) (应该推荐的电影，即看过的电影里评分较高的), ...}
def MAP(user_recommend, test_dataset):
    APs = []
    for usr in test_dataset.keys():
        ap = 0
        if usr not in user_recommend.keys():
            raise KeyError(f'user {usr} has not been predicted')
        movie_order = []
        for movie in test_dataset[usr]:
            if movie not in user_recommend[usr].keys():
                movie_order.append(-1)
            movie_order.append(user_recommend[usr][movie])
        movie_order = np.array(movie_order)
        movie_order[movie_order == -1] = np.max(movie_order) + 1
        movie_order.sort()
        for i in range(len(movie_order)):
            ap += (len(movie_order) - i) / movie_order[i]
        ap /= len(movie_order)
        APs.append(ap)

    return np.mean(APs)
