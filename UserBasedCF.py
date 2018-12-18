import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import json
import random
'''
    基于用户领域的协同过滤算法，
    三种计算用户相似度的方式：
        1） 余弦相似度
        2）IIF
        3）皮尔逊相关系数
'''
class UserBasedCF:
    def __init__(self,fn='ratings_data.txt'):
        self.fn = fn
        self.read_data()
        self.split_data(3, 10)
        #self.calc_sim_pearson()
        self.load_sim_p()
        #self.load_sim_matrix()
        _, self.user_means =  self.compute_sth()

    # 从硬盘读取数据
    def read_data(self):
        with open(self.fn, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            f.close()
        self.data = []
        self.items = set()
        for line in lines:
            if not line[:1].isalnum():
                continue
            nums = line.strip().split()
            # 评分高于三，即认为该用户喜欢此物品
            # nums:{user_id, item_id, record}
            #if nums[0] and int(nums[2]) > 0:
            self.data.append((nums[0], nums[1], int(nums[2])))
            self.items.add(nums[1])

    def visualize_data(self):
        X = []
        y = []
        fig = plt.figure()
        i = 0
        for key, val, _ in self.data:
            i = i + 1
            if i > 100:
                break
            X.append(int(key))
            y.append(len(val))
        plt.plot(X,y, 'r', linewidth = 2)
        plt.show()
        y = pd.DataFrame(y, columns=['counts'])
        y.describe()
        # 查看标记喜欢人数的百分比中位数
        print(np.percentile(y, 99))

    def split_data(self,k, M, seed=43):
        '''
        将数据集划分为训练集和测试集，
        len(训练集) / len(测试集) = 1 / （M - 1）
        :param k:
        :param seed:
        :param M:
        :return:
        '''
        random.seed(seed)
        self.trn_data = {}
        self.tst_data = {}
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                self.tst_data.setdefault(user,{})
                self.tst_data[user][item] = record
            else:
                self.trn_data.setdefault(user, {})
                self.trn_data[user][item] = record

    def calc_sim(self,trn_data):
        '''brute force
        计算训练集中每两个用户的余弦相似度，'''

        self.simMatrix = dict()
        for u0 in trn_data.keys():
            for u1 in trn_data.keys():
                if u0 != u1 and (u0, u1) not in self.simMatrix.keys():
                    self.simMatrix[(u0, u1)] = len(trn_data[u0] & trn_data[u1])
                    self.simMatrix[(u0, u1)] /= math.sqrt(len(trn_data[u0]) * len(trn_data[u1]))
                    self.simMatrix[(u1, u0)] = self.simMatrix[(u0,u1)]

    def compute_sth(self):
        item_users = dict()
        #  计算用户打分平均值
        user_means = dict()
        for user, item in self.trn_data.items():
            nr = 0.0
            sum_r = 0.0
            for i, ur in item.items():
                item_users.setdefault(i, set())
                item_users[i].add(user)
                nr += 1
                sum_r += ur
            user_means.setdefault(user, sum_r / nr)
        return item_users, user_means

    def calc_sim_pearson(self, train=None):
        """根据皮尔逊相关度计算用户相关度"""
        trn_data = train or self.trn_data
        self.simi_pearson = dict()

        item_users, user_means = self.compute_sth()

        # 保存用户u,v共同item
        count = dict()
        for item, users in item_users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, set())
                    count[u][v].add(item)

        for u, related_users in count.items():
            self.simi_pearson.setdefault(u, dict())
            for v, items in related_users.items():
                mole = 0.0
                denomi0 = 0.0
                denomi1 = 0.0
                for item in items:
                    mole += (self.trn_data[u][item] - user_means[u]) * (self.trn_data[v][item] - user_means[v])
                    denomi0 += math.pow(self.trn_data[u][item] - user_means[u], 2)
                    denomi1 += math.pow(self.trn_data[v][item] - user_means[v], 2)
                if denomi0 != 0 and denomi1 != 0:
                    self.simi_pearson.setdefault(u, dict())
                    self.simi_pearson[u][v] = round(mole / math.sqrt(denomi0 * denomi1), 4)

        with open('simMatrix_pearson.json', 'w', encoding='utf-8') as f:
            json.dump(self.simi_pearson, f, ensure_ascii=False)
            f.close()

    def load_sim_p(self):
        with open('simMatrix_pearson.json','r', encoding='utf-8',errors='ignore') as f:
            self.simi_pearson = json.load(f)
            f.close()

    def recommend_p(self, user, train=None, k=10, nitems=10):
        trn_data = train or self.trn_data
        related_users = self.simi_pearson.get(user, {})
        topk_related_users = dict(sorted(related_users.items(), key=lambda x: x[1], reverse=True)[:k])
        rank = dict()
        for ruser, wuv in topk_related_users.items():
            for item, rvi in self.trn_data[ruser].items():
                rank.setdefault(item, self.user_means[user])
                rank[item] += wuv * (rvi - self.user_means[ruser])

        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[:nitems])

    def calc_sim_iif(self, train=None):
        '''计算训练集中每两个用户的余弦相似度，UserCF-IIF
        先计算对同一物品产生过行为的用户列表， N(u)∩N(v) ≠0的用户对(u,v)，
        N(u) 代表用户u喜欢的所有物品'''

        trn_data = train or self.trn_data
        self.simMatrix = dict()
        item_users = dict()
        # 记录对各个物品产生行为的所有用户
        for user, item in trn_data.items():
            for i in item.keys():
                # insert a set in the dictionary if the key is not in the dictionary
                item_users.setdefault(i, set())
                item_users[i].add(user)
        # 记录N(u)∩N(v)
        count = dict()
        N = dict()
        for item, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                for v in users:
                    if u == v: continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1 / math.log(1 + len(users))

        # 计算 similarity matrix between users
        for u, related_users in count.items():
            self.simMatrix.setdefault(u, dict())
            for v, cuv in related_users.items():
                self.simMatrix[u][v] = round(cuv / math.sqrt(N[u] * N[v] * 1.0), 4)

        # 保存矩阵
        with open('simMatrix.json', 'w', encoding='utf-8') as f:
            json.dump(self.simMatrix, f, ensure_ascii=False)
            f.close()

    def load_sim_iif(self):
        with open('simMatrix.json') as data_file:
            self.simMatrix = json.load(data_file)
            data_file.close()

    def recommend(self, user_id, k=50, item_nums=10, train=None):
        '''

        :param user_id: user id
        :param k: 兴趣相似的K个用户
        :param item_nums: 推荐物品数量
        :return:
        '''
        trn_data = train or self.trn_data

        interacted_items = trn_data.get(user_id, {})
        rank = dict()
        related_users = self.simMatrix.get(user_id, {})
        sorted_related_users = sorted(related_users.items(), key=lambda x:x[1], reverse=True)[:k]
        for v, sc in sorted_related_users:
            for i, rvi in trn_data[v].items():
                if i in interacted_items.keys():
                    continue
                rank.setdefault(i, 0)
                rank[i] += sc * int(rvi)
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse = True)[:item_nums])

    # 随机推荐一款用户未产生行为的产品
    def recommend_random(self,user_id, item_nums=10, train=None):
        trn_data = train or self.trn_data
        interacted_items = trn_data.get(user_id, {})
        res = dict()

        N = len(self.items)
        items = list(self.items)

        while len(res) < item_nums:
            idx = random.randint(0,N-1)
            item = items[idx]
            if item not in interacted_items.keys():
                res.setdefault(item, 0.0)
        return res

    # 计算召回率和精确率
    def recall_and_precision(self, train=None, test=None, k=25, item_nums=10, is_random=False):
        trn_data = train or self.trn_data
        tst_data = test or self.tst_data
        hit, recall, precision = 0, 0, 0

        # nuser = len(trn_data)
        # i = 0
        # print('num of users: {0}'.format(nuser))

        for user in trn_data.keys():
            # 打印进度条
            # perc = i / float(nuser)
            # print("Overall percentage: " + str(perc) + " %")
            # print(">" * (100 * int(perc)))

            tu = tst_data.get(user, {})

            if is_random:
                rank = self.recommend_random(user, item_nums=item_nums)
            else:
                rank = self.recommend(user, k=k, item_nums=item_nums)

            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += item_nums

            #i += 1
        return hit / (recall * 1.0), hit / (precision * 1.0)

    # 计算覆盖率
    def coverage(self, train=None, test=None, k=25, item_nums=10, is_random=False):
        trn_data = train or self.trn_data
        tst_data = test or self.tst_data
        all_items = set()
        recommended_items = set()
        for user in trn_data.keys():
            for item in trn_data[user].keys():
                all_items.add(item)

            if is_random:
                rank = self.recommend_random(user, item_nums=item_nums)
            else:
                rank = self.recommend(user, k=k, item_nums=item_nums)

            for item, _ in rank.items():
                recommended_items.add(item)
        return len(recommended_items) / (len(all_items) * 1.0)

    # 计算热度
    def popularity(self, train=None, test=None, k=25, item_nums=10, is_random=False):
        trn_data = train or self.trn_data
        tst_data = test or self.tst_data
        item_popularity = dict()
        for user in trn_data.keys():
            for item, _ in trn_data[user].items():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in trn_data.keys():
            if is_random:
                rank = self.recommend_random(user, item_nums=item_nums)
            else:
                rank = self.recommend(user, k=k, item_nums=item_nums)
            for item, _ in rank.items():
                ret += math.log(1 + item_popularity.get(item, 0))
                n += 1
        return ret / (n * 1.0)

def testUserBasedCF():
    ubCF = UserBasedCF()
    print("     recall  precision  coverage  popularity")
    ks = [5,10,20,40,80,160]
    for k in ks:
        recall, precision = ubCF.recall_and_precision(k=k)
        coverage = ubCF.coverage(k=k)
        popularity = ubCF.popularity(k=k)
        print("k={:d} {:.4f}%  {:.4f}%  {:.4f}%  {:.4f}".format(k, recall*100, precision*100, coverage*100, popularity))
        recall, precision = ubCF.recall_and_precision(k=k, is_random=True)
        coverage = ubCF.coverage(k=k, is_random=True)
        popularity = ubCF.popularity(k=k, is_random=True)
        print("k={:d} {:.4f}%  {:.4f}%  {:.4f}%  {:.4f} -random".format(k, recall*100, precision*100, coverage*100, popularity))

def testPearson():
    ubCF = UserBasedCF()
    rank = ubCF.recommend_p('123')
    print(rank)

if __name__ == '__main__':
    #testUserBasedCF()
    testPearson()


