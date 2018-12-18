import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import json

class ItemBasedCF:
    def __init__(self, fn='ratings_data.txt'):
        self.fn = fn
        self.read_data()
        self.split_data(3, 10)
        import os
        if os.path.exists("simiMatrix_item.json"):
            self.load_sim_matrix()
        else:
            self.calc_simi()

    def read_data(self):
        with open(self.fn, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            f.close()
        self.data = []
        for line in lines:
            if not line[:1].isalnum():
                continue
            nums = line.strip().split()
            # 评分高于三，即认为该用户喜欢此物品
            # nums:{user_id, item_id, record}
            # if nums[0] and int(nums[2]) > 0:
            self.data.append((nums[0], nums[1], nums[2]))

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
        plt.plot(X, y, 'r', linewidth=2)
        plt.show()
        y = pd.DataFrame(y, columns=['counts'])
        y.describe()
        # 查看标记喜欢人数的百分比中位数
        print(np.percentile(y, 99))

    def split_data(self, k, M, seed=43):
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
                self.tst_data.setdefault(user, {})
                self.tst_data[user][item] = record
            else:
                self.trn_data.setdefault(user, {})
                self.trn_data[user][item] = record

    def calc_simi(self, train=None):
        '''
        基于物品计算
        :param train:
        :return:
        '''
        trn_data = train or self.trn_data
        self.simiMatrix = dict()

        count = dict()
        N = dict()
        for user, items in trn_data.items():
            for i in items.keys():
                N.setdefault(i, 0)
                N[i] += 1
                for j in items.keys():
                    if i == j:
                        continue
                    count.setdefault(i, {})
                    count[i].setdefault(j, 0)
                    count[i][j] += 1

        for i, related_items in count.items():
            self.simiMatrix.setdefault(i, {})
            for j, cuv in related_items.items():
                self.simiMatrix[i].setdefault(j, 0)
                self.simiMatrix[i][j] = round(cuv / math.sqrt(N[i] * N[j] * 1.0), 4)

        with open("simiMatrix_item.json",'w', encoding="utf-8") as f:
            json.dump(self.simiMatrix, f, ensure_ascii=False)
            f.close()

    def recommend(self, user, train=None, k=10, nitems=10):
        trn_data = train or self.trn_data
        ru = trn_data.get(user, {})
        rank = dict()
        for i, rec in ru.items():
            for j, wj in sorted(self.simiMatrix.get(i,{}).items(), key=lambda x:x[1], reverse=True)[:k]:
                if j in ru:
                    continue
                rank.setdefault(j, 0)
                rank[j] += int(rec) * wj
        rs = dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[:nitems])
        return rs

    def load_sim_matrix(self):
        with open('simiMatrix_item.json', encoding='utf-8') as f:
            self.simiMatrix = json.load(f)
            f.close()

    def recall_precision(self,train=None, test=None, k=10, nitems=10):
        tst_data = test or self.tst_data
        trn_data = train or self.trn_data
        hit = 0
        recall = 0
        precision = 0
        # nuser = len(trn_data)
        # print(nuser)
        # i = 0
        for user in trn_data.keys():
            #打印进度条
            # perc = i / float(nuser)
            # print("Overall percentage: " + str(perc))
            # print(">" * (100 * int(perc)))


            tu = tst_data.get(user, {})
            recommended_items = self.recommend(user, k=k, nitems=nitems)
            for item in recommended_items.keys():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitems

            #i += 1

        return hit / (recall * 1.0), hit / (precision * 1.0)

    def coverage(self, train=None, test=None, k=10, nitems=10):
        trn_data = train or self.trn_data
        tst_data = test or self.tst_data
        all_items = set()
        recommended_items = set()
        for user in trn_data.keys():
            for item in trn_data[user].keys():
                all_items.add(item)
            rank = self.recommend(user, k=k, nitems=nitems)
            for item in rank.keys():
                recommended_items.add(item)

        return len(recommended_items) / float(len(all_items))

    def popularity(self, train=None, test=None, k=10, nitems=10):
        '''
        计算推荐物品的平均热门程度
        :param train:
        :param test:
        :param k:
        :param nitems:
        :return:
        '''
        trn_data = train or self.trn_data
        tst_data = test or self.tst_data

        ret = 0
        n = 0
        item_popularity = dict()

        for user in trn_data.keys():
            for item, _ in trn_data[user].items():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1

        for user in trn_data.keys():
            rank = self.recommend(user, k=k,nitems=nitems)
            for item in rank.keys():
                ret += math.log(1 + item_popularity[item])
                n += 1

        return ret / (n * 1.0)



def testItemBasedCF():
    ibcf = ItemBasedCF()
    print(ibcf.recommend("126"))
    print("recall  precision  coverage  popularity")
    ks = [5, 10]
    for k in ks:
        recall, precision = ibcf.recall_precision(k=k)
        coverage = ibcf.coverage(k=k)
        popularity = ibcf.popularity(k=k)
        print("{:.4f}  {:.4f}     {:.4f}    {:.4f}".format(recall, precision, coverage, popularity))

if __name__ == "__main__":
    testItemBasedCF()
