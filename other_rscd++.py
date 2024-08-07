import numpy as np
import math

# user number
n = 943
# item number
m = 1682
r_ui = np.zeros((n, m), int)  # rating matrix
y_ui = np.zeros((n, m), int)
# observed items by user u
Ius_ = np.zeros((n, m), int)
# The tradeoff parameters
au = av = aw = Bv = Bu = 0.01
# The learning rate
lr = 0.01
# The number of latent dimensions
d = 20
# Iteration number
T = 100
with open("datasets/ml-100k/ua.base.explicit", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        r_ui[int(user) - 1][int(item) - 1] = int(rating)
        if rating:
            y_ui[int(user) - 1][int(item) - 1] = 1
with open("datasets/ml-100k/ua.base.implicit", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        Ius_[int(user) - 1][int(item) - 1] = int(rating)

rm_ = np.zeros((n, m), int)
rm_p = np.zeros((n, m), float)  # 保存预测值
with open("datasets/ml-100k/ua.test", 'r') as k:
    for line in k.readlines():
        user, item, rating, _ = line.split('	')
        rm_[int(user) - 1][int(item) - 1] = int(rating)
r = np.count_nonzero(rm_)

# number of observed ratings  p = |R|
p = np.sum(y_ui)


def initial():
    # user-specific latent feature vector
    Uuk = np.zeros(d, float)
    # item-specific latent feature vector
    Vik = np.zeros(d, float)
    # item-specific latent feature vector
    Wik = np.zeros(d, float)
    u = np.sum(r_ui * y_ui) / np.sum(y_ui)
    bu = np.zeros(n, float)
    for i in range(n):
        if np.sum(y_ui[i]) == 0:
            continue
        bu[i] = np.sum(y_ui[i] * (r_ui[i] - u)) / np.sum(y_ui[i])
    bi = np.zeros(m)
    for i in range(m):
        if np.sum(y_ui[:, i]) == 0:
            continue
        bi[i] = np.sum(y_ui[:, i] * (r_ui[:, i] - u)) / np.sum(y_ui[:, i])

    for i in range(d):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        r3 = np.random.uniform(0, 1)
        Uuk[i] = (r1 - 0.5) * 0.01
        Vik[i] = (r2 - 0.5) * 0.01
        Wik[i] = (r3 - 0.5) * 0.01
    Uuk = np.tile(Uuk, (n, 1))  # n*d
    Vik = np.tile(Vik, (m, 1))
    Wik = np.tile(Wik, (m, 1))
    return u, bu, bi, Uuk, Vik, Wik


# 计算预测后值

s1 = 0
for i in range(n):
    for j in range(m):
        if Ius_[i][j]:
            s1 += 1
s1 = 1. / math.sqrt(math.fabs(s1))


def predict(i, j):
    u1 = np.zeros(d, float)
    Iu_ = Ius_[i]
    # 这个地方是新加入的矩阵
    for x in range(len(Iu_)):
        if Iu_[x]:
            u1 += Wik[x]
    Uu_ = u1 / s1
    return bu[i] + bi[j] + np.dot(Uuk[i], Vik[j].T) + u + np.dot(Uu_, Vik[j].T)


def gradient(i, j):
    e_ui = r_ui[i][j] - predict(i, j)
    __u = -e_ui
    __bu = -e_ui + Bu * bu[i]
    __bi = -e_ui + Bv * bi[j]
    __Uu = -e_ui * Vik[j] + au * Uuk[i]
    __Vi = -e_ui * Uuk[i] + av * Vik[j]
    __Wi = -e_ui * s1 * Vik[j] + aw * Wik[j]
    return __u, __bu, __bi, __Uu, __Vi, __Wi


def MAE(r1, r2, r_):
    return np.sum(np.abs(r1 - r2)) / r_


def RMSE(r1, r2, r_):
    return math.sqrt(np.sum((r1 - r2) * (r1 - r2)) / r_)


def model_test():
    for i in range(n):
        for j in range(m):
            if rm_[i][j] != 0:
                rm_p[i][j] = predict(i, j)
    print("MAE:%s ,RMSE:%s" % (MAE(rm_, rm_p, r), RMSE(rm_, rm_p, r)))


if __name__ == "__main__":
    u, bu, bi, Uuk, Vik, Wik = initial()
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
            _u, _bu, _bi, _Uu, _Vi, _Wi = gradient(user_u, user_i)
            # update
            u = u - lr * _u
            bu[user_u] = bu[user_u] - lr * _bu
            bi[user_i] = bi[user_i] - lr * _bi
            Uuk[user_u] = Uuk[user_u] - lr * _Uu
            Vik[user_i] = Vik[user_i] - lr * _Vi
            Wik[user_i] = Wik[user_i] - lr * _Wi
        model_test()
        lr = lr * 0.9
    # print(u, bu, bi, Uuk, Vik)