
# 协同过滤推荐
协同过滤（Collaborative Filtering）算法基于Kth领域，主要利用行为的相似度计算兴趣的相似度
## 基于用户的协同过滤推荐（User based)

### 用户相似度计算

计算规则：

1）给定用户u，v，其之间相似度计算，采用余弦相似度， 做归一化处理
N（u）- 用户u有过正反馈的物品集合， N（v）- 用户v有过正反馈的物品集合, Wu - 用户
u，v之间的相似度，该规则称UserCF算法:

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/ub_1.png)


2）给定用户u，v, 如果其都对冷门物品采取过正反馈，则更能说明他们兴趣的相似度，
也可以理解为二者喜欢的共同商品越热门，其带来的相似度增加值越低，引用来至John S.Breese，
该规则称UserCF-IIF算法:

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/ub_2.png)

N(i) - 喜欢商品i的用户集合，  log函数惩罚了用户u和用户v共同兴趣列表中热门物品对他们相似度的影响

3）Adjusted Cosine Similarity， 修正余弦相似度，
由于余弦相似度没有考虑不同用户的评分尺度的问题，修正的余弦相似度通过减去用户对物品的平均评分来做中心化处理

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/Adjusted_cosine_similarity.jpg)

Ii和Ij分别指经过用户i，j评分的项目集合，Iij指用户i，j共同评分的集合

4）Pearson Correlation Coefficient，皮尔逊相关系数，与修正余弦相似度类似， 不同的时，其中心化的方式不同

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/Pearson_correlation_cofficient.jpg)


### 用户u对物品i的兴趣:

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/ub_3.png)

S(u,k) - 与用户u的K个最邻近的用户集合
## 基于物品的协同过滤推荐 （Item based）

物品i，j之间的相似度计算：

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/ib_1.png)

N(i) - 喜欢物品i的用户数

|N(i)  ∩N(j))| - 同时喜欢物品i， j的用户数

用户u对物品i的兴趣:

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/ib_2.png)

S(i, k) - 与物品i最邻近的k个物品

Wij - 物品i，j相似度

Rui - 用户u对物品i的打分
## 评估

### 召回率（Recall）:

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/recall.png)

R(u) - 针对用户u所推荐的物品集合

T(u) - 测试集中用户u所产生行为的物品集合

### 精准率(Precision):

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/precision.png)

### 覆盖率(Coverage):
定义：推荐系统所推荐出来的物品占总物品集合的比例

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/coverage.png)

U – 用户集合
R(u) – 推荐系统给用户推荐的长度为N的物品列表
I – 训练集所有物品集合

### 平均热门程度（Average Popularity）：
推荐结果的平均热门程度

![](https://raw.githubusercontent.com/Neoooou/Recommendation-System/master/img/popularity.png)

Item_pop(i)： 为该物品作出行为的用户数

#运行结果(userCF-IIF)

  K  |recall | precision | coverage | popularity
  ---|-------|-----------|----------|----------
  5  |1.3093%| 0.1974%   | 32.6379% | 3.0822
  10 |1.7614%|  0.2656%  | 33.4529% | 3.1857
  20 |2.2270%| 0.3357%   |  32.4566%| 3.3585
  40 |2.8959%| 0.4366%   |  29.9789%|  3.7459
  80 |3.8674%| 0.5830%   |  25.8953%|  4.3813
  160|4.6640%| 0.7032%   | 20.9408% |  5.0214
  
  与随机推荐比较(k=40)：
  
   ~ | Recall| precision |coverage|popularity
  ---|-------|-----------|--------|----------
  Random|0.0084%|0.0013%|99.3285%|1.0926
  UserCF-IIF|2.8959%|0.4366%|29.9789%|3.7459

# 测试
    运行命令：
        set FLASK_APP=Ser.py
        flask run
    
    以post方式发送http请求http://localhost:5000/get_recmd，请求主体文本以JSON格式{'user_id':"xxx",
    "item_nums":N}
    返回结果为[recommended_item1, recommended_item2,....]
