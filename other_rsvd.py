import numpy as np
import math

user_num = 943
item_num = 1682
# Iteration number
T = 100
# The tradeoff parameters
au = av = Bu = Bv = 0.01
# The number of latent dimensions
d = 20

r_ui = np.zeros((user_num, item_num), int)  # rating matrix
y_ui = np.zeros((user_num, item_num), int)

with open("datasets/ml-100k/u1.base", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        r_ui[int(user) - 1][int(item) - 1] = int(rating)
        if rating:
            y_ui[int(user) - 1][int(item) - 1] = 1
# number of observed ratings
p = np.sum(y_ui)

rm_ = np.zeros((user_num, item_num), int)
rm_p = np.zeros((user_num, item_num), float)  # 保存预测值
with open("datasets/ml-100k/u1.test", 'r') as k:
    for line in k.readlines():
        user, item, rating, _ = line.split('	')
        rm_[int(user) - 1][int(item) - 1] = int(rating)
r = np.count_nonzero(rm_)


def initial():
    # user-specific latent feature vector
    Uuk = np.zeros(d, float)
    # item-specific latent feature vector
    Vik = np.zeros(d, float)
    u = np.sum(r_ui * y_ui) / np.sum(y_ui)  # 单一值
    bu = np.zeros(user_num, float)
    for i in range(user_num):
        if np.sum(y_ui[i]) == 0:
            continue
        bu[i] = np.sum(y_ui[i] * (r_ui[i] - u)) / np.sum(y_ui[i])
    bi = np.zeros(item_num)
    for i in range(item_num):
        if np.sum(y_ui[:, i]) == 0:
            continue
        bi[i] = np.sum(y_ui[:, i] * (r_ui[:, i] - u)) / np.sum(y_ui[:, i])

    for i in range(d):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        Uuk[i] = (r1 - 0.5) * 0.01
        Vik[i] = (r2 - 0.5) * 0.01
    Uuk = np.tile(Uuk, (user_num, 1))
    Vik = np.tile(Vik, (item_num, 1))
    return u, bu, bi, Uuk, Vik


# 计算预测后值
def predict(i, j):
    return bu[i] + bi[j] + np.dot(Uuk[i], Vik[j].T) + u


def gradient():
    e_ui = r_ui[user_u][user_i] - predict(user_u, user_i)
    __u = -e_ui
    __bu = -e_ui + Bu * bu[user_u]
    __bi = -e_ui + Bv * bi[user_i]
    __Uu = -e_ui * Vik[user_i] + au * Uuk[user_u]
    __Vi = -e_ui * Uuk[user_u] + av * Vik[user_i]
    return __u, __bu, __bi, __Uu, __Vi


def MAE(r1, r2, r_):
    return np.sum(np.abs(r1 - r2)) / r_


def RMSE(r1, r2, r_):
    return math.sqrt(np.sum((r1 - r2) * (r1 - r2)) / r_)


def model_test():
    for i in range(user_num):
        for j in range(item_num):
            if rm_[i][j] != 0:
                rm_p[i][j] = predict(i, j)
    print("MAE:%s ,RMSE:%s" % (MAE(rm_, rm_p, r), RMSE(rm_, rm_p, r)))


if __name__ == "__main__":
    u, bu, bi, Uuk, Vik = initial()
    # The learning rate
    lr = 0.01
    # 评价矩阵中非0索引
    non_zero_indices = np.where(r_ui != 0)
    len_index = len(non_zero_indices[0])
    for i in range(T):
        for j in range(1, p + 1):
            i_k = np.random.randint(0, len_index)
            user_u = non_zero_indices[0][i_k]
            user_i = non_zero_indices[1][i_k]
            _u, _bu, _bi, _Uu, _Vi = gradient()
            # update
            u = u - lr * _u
            bu[user_u] = bu[user_u] - lr * _bu
            bi[user_i] = bi[user_i] - lr * _bi
            Uuk[user_u] = Uuk[user_u] - lr * _Uu
            Vik[user_i] = Vik[user_i] - lr * _Vi
        model_test()
        lr = lr * 0.9
    print(u, bu, bi, Uuk, Vik)