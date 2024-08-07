import pandas as pd
import numpy as np 
import math
import codecs
import json
import random
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

random.seed(2024)
train_data = pd.read_csv("datasets/ml-100k/u1.base",sep='\t',names=['uid','iid','rating'],usecols=[0,1,2],header=None)
test_data = pd.read_csv("datasets/ml-100k/u1.test",sep='\t',names=['uid','iid','rating'],usecols=[0,1,2],header=None)
user_set = set(train_data['uid'])
item_set = set(train_data['iid'])
user_len = 943
item_len = 1682
rating = train_data.values
# 评分的集合
rating_set = {1,2,3,4,5}

# PMF 模型
class MPC:
    # PMF 模型初始化，已经设置默认参数
    def __init__(self, user_set, item_set, record_list, dimensions=20, learning_rate=0.01, alpha_user=0.001, alpha_item=0.001, alpha_context=0.001, beta_user=0.001, beta_item=0.001):
        # 创建PMF时，表示用户id的set集合。调用vector_initialize函数后，表示用户的特征矩阵 {用户id：用户特征向量，...}
        self.users = user_set
        # 同上
        self.items = item_set
        # 用户评过分的物品集合,具体到每个分数 {1:{"1":[2,3,4],...,"5":[5,19,..]},...}
        self.user_rated_items = {}
        # 偏置向量
        self.user_bias = 0
        self.item_bias = 0
        self.avg_rating = 0
        # 训练集中的记录列表
        self.records = record_list
        # 用户和物品的特征维度，默认为20
        self.dimensions = dimensions
        # 学习率，默认为0.01
        self.learning_rate = learning_rate
        # 用户正则化的超参数，默认为0.01
        self.alpha_user = alpha_user
        # 物品正则化的超参数，默认为0.01
        self.alpha_item = alpha_item
        # 用户偏好上下文特征向量正则化的超参数，默认为0.01
        self.alpha_context = alpha_context
        # 用户偏差的正则化参数
        self.beta_user = beta_user
        # 物品偏差的正则化参数
        self.beta_item = beta_item
        # 训练过程中的损失
        self.loss = 0

    # 初始化用户特征和物品特征
    def vector_initialize(self):
        self.avg_rating = self.records[:,2].mean()
        # 用户和物品的特征使用字典来保存，Key是ID，Value是相应的特征向量
        user_bias_dict = {}
        item_bias_dict = {}
        # 用户特征初始化
        user_feature_matrix = np.random.rand(user_len, self.dimensions).astype(np.float32)
        user_feature_matrix = (user_feature_matrix - 0.5) * 0.01
        item_feature_matrix = np.random.rand(item_len, self.dimensions).astype(np.float32)
        item_feature_matrix = (item_feature_matrix - 0.5) * 0.01
        item_feature_rating_matrices = []
        for i in rating_set:
            temp_matrix = np.random.rand(item_len, self.dimensions).astype(np.float32)
            temp_matrix = (item_feature_matrix - 0.5) * 0.01
            item_feature_rating_matrices.append(temp_matrix)
        self.users = dict(zip(
            list(range(1, user_len+1)),
            user_feature_matrix
        ))
        self.items = dict(zip(
            list(range(1, item_len+1)),
            item_feature_matrix
        ))
        self.item_rating_contexts = item_feature_rating_matrices

        for user_id in self.users:
            indexs = (self.records[:,0] == int(user_id))
            if (indexs == False).all():
                continue
            user_bias_dict[user_id] = (self.records[indexs][:,2] - self.avg_rating).mean()
            self.user_rated_items[user_id] = {}
            # 需要获取每个用户不同评分的物品集合的index
            for r in rating_set:
                temp_rating_indexs = indexs & (rating[:,2] == r)
                self.user_rated_items[user_id][r] = set(self.records[temp_rating_indexs][:,1])
        # 物品特征初始化
        for item_id in self.items:
            indexs = (self.records[:,1] == int(item_id))
            # 训练集中有些物品没有出现
            if (indexs == False).all():
                continue
            item_bias_dict[item_id] = (self.records[indexs][:,2] - self.avg_rating).mean()
        
        self.user_bias = user_bias_dict
        self.item_bias = item_bias_dict

    # 使用随机梯度下降方法训练用户和物品的特征
    def train(self, epochs, test_data):
        # 迭代次数
        rmse_list = []
        mae_list = []
        for epoch in range(epochs):
            # 每次迭代开始，将模型的属性loss置0
            self.loss = 0
            # 遍历评分记录
            start_time = time.time()
            for record in self.records:
                sample = self.records[random.randint(0,len(self.records)-1)]

                user_id = sample[0]
                item_id = sample[1]
                # 该记录的用户对物品的评分
                rating = int(sample[2])
                # 该记录的用户特征向量
                user_vector = self.users[user_id]
                # 该记录的物品特征向量
                item_vector = self.items[item_id]
                bias_u = self.user_bias[user_id]
                bias_i = self.item_bias[item_id]
                # 用户u的上下文偏好的特征向量
                # 归一化的分母值
                user_multi_class_vector = 0
                user_rated_items_except_item = self.user_rated_items[user_id].copy()
                for r,item_ids in user_rated_items_except_item.items():
                    item_ids.discard(item_id) 
                    norm_len = len(item_ids)
                    if norm_len == 0:
                        continue
                    # 找到下标，从0开始，所以-1
                    indexs = np.array(list(item_ids)) - 1
                    try:
                        user_multi_class_vector += (np.sum(self.item_rating_contexts[r-1][indexs], axis=0)) / math.sqrt(norm_len)
                    except:
                        print(indexs)
                # 预测值
                predict_value = self.predict_post_process(
                    np.dot(
                        (
                            self.users[user_id] + 
                            user_multi_class_vector
                        ), 
                        self.items[item_id]
                    ) + 
                    self.user_bias[user_id] + 
                    self.item_bias[item_id] + 
                    self.avg_rating
                )
                # 计算损失
                # error = self.loss_function(user_id, item_id, rating, predict_value, user_rated_items_except_item)
                # 损失累加
                # self.loss += error
                # 全局平均的梯度
                grad_avg_rating = predict_value - rating
                # 用户偏差的梯度
                grad_bias_user = predict_value - rating + self.beta_user * bias_u
                # 用户偏差的梯度
                grad_bias_item = predict_value - rating + self.beta_item * bias_i
                # 计算该用户特征向量的梯度
                grad_user = (predict_value - rating) * item_vector + self.alpha_user * user_vector
                # 计算该物品特征向量的梯度
                grad_item = (predict_value - rating) * (user_vector + user_multi_class_vector) + self.alpha_item * item_vector
                # 物品的不同评分矩阵的特征向量的梯度
                for r,item_ids in user_rated_items_except_item.items():
                    item_ids.discard(item_id) 
                    # 找到下标，从0开始，所以-1
                    norm_len = len(item_ids)
                    if norm_len == 0:
                        continue
                    indexs = np.array(list(item_ids)) - 1   
                    
                    grad_item_ratings = (predict_value - rating) * item_vector / norm_len + self.alpha_context * self.item_rating_contexts[r-1][indexs]
                    self.item_rating_contexts[r-1][indexs] = self.item_rating_contexts[r-1][indexs] - self.learning_rate * grad_item_ratings
                # 根据梯度对特征向量进行更新
                self.avg_rating -= self.learning_rate * grad_avg_rating
                self.user_bias[user_id] -= self.learning_rate * grad_bias_user
                self.item_bias[item_id] -= self.learning_rate * grad_bias_item
                self.users[user_id] -= self.learning_rate * grad_user
                self.items[item_id] -= self.learning_rate * grad_item
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"运行时间: {elapsed_time:.7f} 秒")
            # 每迭代完一次，学习率降低
            self.learning_rate = self.learning_rate * 0.9
            predict, ground_value = self.test(test_data)
            rmse = math.sqrt(mean_squared_error(predict,ground_value))
            mae = mean_absolute_error(predict,ground_value)
            rmse_list.append(rmse)
            mae_list.append(mae)
            print("epoch: ", epoch)
            print('RMSE={}'.format(rmse))
            print('MAE={}'.format(mae))
            # 打印每次迭代的损失
            # print("epoch: ", epoch, "loss:", self.loss)

        self.draw(epochs, rmse_list, mae_list)

        # 训练完之后，将用户特征向量进行保存
        with codecs.open("pureResult/user_vector.json", "w") as f1:
            for u in self.users.keys():
                self.users[u] = self.users[u].tolist()
            json.dump(self.users, f1)
        # 将物品特征向量进行保存
        with codecs.open("pureResult/item_vector.json", "w") as f2:
            for i in self.items.keys():
                self.items[i] = self.items[i].tolist()
            json.dump(self.items, f2)
    # 损失函数定义
    def loss_function(self, user_id, item_id, rating, predict_value, user_rated_items_except_item):      # TODO 这个最后一个计算比较耗费时间
        return 0.5 * math.pow((rating - predict_value), 2) + \
               0.5 * self.alpha_user * math.pow(np.linalg.norm(self.users[user_id], ord=2), 2) + \
               0.5 * self.alpha_item * math.pow(np.linalg.norm(self.items[item_id], ord=2), 2) + \
               0.5 * self.beta_user * math.pow(self.user_bias[user_id], 2) + \
               0.5 * self.beta_item * math.pow(self.item_bias[item_id], 2) + \
               0.5 * self.alpha_context * np.sum([np.sum(
                    np.square(
                       np.linalg.norm(
                            self.item_rating_contexts[r-1][user_rated_items_except_item],
                            axis=1
                        ) 
                    )
               ) for r in rating_set])


    def predict(self, uid, iid):
        # 测试集中，有些电影在训练集中没有出现
        if iid not in self.item_bias.keys():
            return self.avg_rating
        user_multi_class_vector = 0
        user_rated_items_except_item = self.user_rated_items[uid].copy()
        for r,items in user_rated_items_except_item.items():
            items.discard(iid) 
            # 找到下标，从0开始，所以-1
            norm_len = len(items)
            if norm_len == 0:
                continue
            user_rated_items_except_item = np.array(list(user_rated_items_except_item)) - 1   
        
            user_multi_class_vector += (np.sum(self.item_rating_contexts[r-1][user_rated_items_except_item], axis=0)) / math.sqrt(norm_len)

        return self.predict_post_process(self.avg_rating + self.user_bias[uid] + self.item_bias[iid] + np.dot((self.users[uid] + user_multi_class_vector), self.items[iid]))

    def predict_post_process(self, rating):
        if rating > 5:
            return 5
        elif rating < 1:
            return 1
        return rating
    
    def test(self, test_data):
        predict_values = []
        for i,row in test_data.iterrows():
            rating = row['rating']
            uid = row['uid']
            iid = row['iid']
            predict_value = self.predict(uid, iid)
            predict_values.append(predict_value)
        return predict_values, list(test_data['rating'].values)


    def draw(self, epochs, rmse_list, mae_list):
        x = range(1, epochs + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(x, rmse_list)
        axs[0].set_title('RMSE={}'.format(round(min(rmse_list),4)))   
        axs[1].plot(x, mae_list)
        axs[1].set_title('MAE={}'.format(round(min(mae_list),4)))
        plt.subplots_adjust(hspace=1.0)
        plt.savefig('image/MF-MPC_epoch={}&dimensions={}.png'.format(epochs,20))


if __name__ == "__main__":
    model = MPC(user_set, item_set, rating)
    start_time = time.time()
    model.vector_initialize()
    print("开始训练")
    model.train(100, test_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time:.2f} 秒")