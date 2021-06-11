

# pyspark读写dataframe
import findspark
findspark.init()

from pyspark.sql import SparkSession
# 创建SparkSession
spark=SparkSession         .builder         .appName('Titanic')         .getOrCreate()




# 读取训练集，带有header，自动推断字段类型
df = spark.read.csv("./train.csv", header=True, inferSchema=True).cache()
# 创建临时表train
df.createOrReplaceTempView("train")
df.show(10)


# ## EDA探索



from pyspark.sql import Row
from pyspark.sql.functions import *
# 输出schema，dataframe的数据结构信息
df.printSchema()




# 对Age字段进行描述统计
df.describe(['age']).show()





# 缺失值统计, Spark SQL类型转换使用cast, col函数将字符串转换为column对象
df.select(*(
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns)).show()





# 使用"""定义多行字符串
query = """
SELECT Embarked, count(PassengerId) as count
FROM train
WHERE Survived = 1
GROUP BY Embarked
"""
spark.sql(query).show()


# ## 数据预处理




from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
# cabin字段缺失值多，去掉该字段
df = df.drop('cabin')
before = df.select('age').where('age is null').count()
print("Age字段缺失个数（处理前）: {}".format(before))
# 使用df.na处理缺失值
test = df.na.drop(subset=["age"])
after = test.select('age').where('age is null').count()
print("Age字段缺失个数（处理后）: {}".format(after))





# 按照survived字段统计
df.groupBy('survived').count().show()

# 按照survived的一定比例进行采样
sample_df = df.sampleBy('survived', fractions={0: 0.1, 1: 0.5}, seed=0)
sample_df.groupBy('survived').count().show()





# 添加字段len_name，代表 乘客name的长度
str_length = udf(lambda x: len(x), IntegerType())
# 使用withColumn添加字段
df = df.withColumn('len_name', str_length(df['name']))
df.select('name', 'len_name').show(5)





# 将类别变量 转化为数值
def embarked_to_int(embarked):
    if embarked == 'C': return 1
    if embarked == 'Q': return 2
    if embarked == 'S': return 3    
    return 0

# 使用udf，定义函数，将类别变量 转化为数值，使用Spark ML中StringIndexer，结果也是一样的
embarked_to_int = udf(embarked_to_int, IntegerType())
# 添加embarked_index字段
df = df.withColumn('embarked_index', embarked_to_int(df['embarked']))
df.select('embarked', 'embarked_index').show(5)




# 计算各列的均值
mean = df.agg(*(mean(c).alias(c) for c in df.columns))
# 字典数据持久化
meaninfo = mean.first().asDict()
print(meaninfo)
# 填充
df = df.fillna(meaninfo["Age"])





# 将sex字段进行数值编码
df.select('sex', 
    when(df['sex'] == 'male', 0).otherwise(1).alias('sex_ix')).show(5)


# ## 数据抽取，转换与特征选择




from pyspark.ml.feature import StringIndexer, VectorAssembler
# StringIndexer将一组字符型标签编码成一组标签索引
df = StringIndexer(inputCol='Sex', outputCol='sex_index').fit(df).transform(df)
df.select('Sex', 'sex_index').show(5)





# 将指定的多个列进行合并，方便后续的ML计算
inputCols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'embarked_index', 'sex_index', 'len_name']
#创建features列，使用VectorAssembler将给定列列表组合成单个向量列
assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
train = assembler.transform(df).select('PassengerId', col('Survived').alias('label'), 'features')





train.show(5)


# ## 模型训练与预测




# 使用随机森林
from pyspark.ml.classification import RandomForestClassifier
# 将数据集切分为80%训练集，20%测试集
splits = train.randomSplit([0.8, 0.2])
train = splits[0].cache()
test = splits[1].cache()

# cacheNodeIds: 是否缓存节点ID
model = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    cacheNodeIds=True)
# 使用train进行训练，test进行预测
predict = model.fit(train).transform(test)
predict.show(5)


# ## 模型评估




from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
"""
    二分类评估：BinaryClassificationEvaluator
    多分类评估：MulticlassClassificationEvaluator
    回归评估：RegressionEvaluator
    聚类评估：ClusteringEvaluator
"""
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", 
    labelCol="label", 
    metricName="accuracy")
print(evaluator.evaluate(predict))

evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='prediction',
    labelCol='label',
    metricName='areaUnderROC')
print(evaluator.evaluate(predict))




