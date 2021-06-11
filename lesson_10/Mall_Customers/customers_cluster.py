


import findspark
findspark.init()

from pyspark.sql import SparkSession
# 创建SparkSession
spark=SparkSession         .builder         .appName('mall_customer')         .getOrCreate()





# 数据加载
df = spark.read.csv('./Mall_Customers.csv', header=True, inferSchema=True)





# 列名修改
df = df.withColumnRenamed('Annual Income (k$)', 'Income').withColumnRenamed('Spending Score (1-100)', 'Spend')
df.show(5)





# 查看是否有缺失值
df.toPandas().isnull().sum()





# 将指定列 组合成 单个列
from pyspark.ml.feature import VectorAssembler
vecAss = VectorAssembler(inputCols = df.columns[3:], outputCol = 'features')
df_km = vecAss.transform(df).select('CustomerID', 'features')
df_km.show(5)




# 将数据转换为pandas格式
df_pd = df.toPandas()
df_pd.head()





import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20, 10))
sns.scatterplot(x='Income', y='Spend', data=df_pd)





# 使用KMeans进行聚类
from pyspark.ml.clustering import KMeans

cost = list(range(2,20))
for k in range(2, 20):
    kmeans = KMeans(k=k, seed=1)
    km_model = kmeans.fit(df_km)
    # computeCost:计算输入点与其对应的聚类中心之间的平方距离之和。
    cost[k-2] = km_model.computeCost(df_km)





plt.figure(figsize=(20,10))
plt.xlabel('k')
plt.ylabel('cost')
plt.plot(range(2,20), cost)





kmeans = KMeans(k=5, seed=1)
km_model = kmeans.fit(df_km)
centers = km_model.clusterCenters()
centers





transformed = km_model.transform(df_km).select('CustomerID', 'prediction')
transformed.show(5)





# 得到预测结果
df_pred = df.join(transformed, 'CustomerID')
df_pred.show(5)




df_pd = df_pred.toPandas()
plt.figure(figsize=(20, 10))
sns.scatterplot(x='Income', y='Spend', data=df_pd)




spark.stop()
