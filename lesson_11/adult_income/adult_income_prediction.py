#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pyspark读写dataframe
import findspark
findspark.init()

from pyspark.sql import SparkSession
# 创建SparkSession
spark=SparkSession         .builder         .appName('adult')         .getOrCreate()


# In[2]:


# 文件带有表头
df = spark.read.csv('adult.csv', inferSchema = True, header=True)
df.show(3)


# In[3]:


cols = df.columns


# In[4]:


df.printSchema()


# In[5]:


# 查看dtypes
df.dtypes


# In[6]:


# 筛选dtypes为string的变量
cat_features = [item[0] for item in df.dtypes if item[1]=='string']
cat_features


# In[8]:


# 需要删除 income列，否则标签泄露
cat_features.remove('income')
cat_features


# In[9]:


num_features = [item[0] for item in df.dtypes if item[1]!='string']
num_features


# In[10]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler

stages = []

"""
    将分类特征进行OneHot编码
    运用StringIndexer 和 OneHotEncoderEstimator处理categorical columns 
    声明stages的顺序和内容
"""
for col in cat_features:
    # 字符串转成索引
    string_index = StringIndexer(inputCol = col, outputCol = col + 'Index')
    # 转换为OneHot编码
    encoder = OneHotEncoderEstimator(inputCols=[string_index.getOutputCol()], outputCols=[col + "_one_hot"])
    # 将每个字段的转换方式 放到stages中
    stages += [string_index, encoder]


# In[11]:


# 将income转换为索引
label_string_index = StringIndexer(inputCol = 'income', outputCol = 'label')
# 添加到stages中
stages += [label_string_index]


# In[12]:


# 类别变量 + 数值变量
assembler_cols = [c + "_one_hot" for c in cat_features] + num_features
assembler = VectorAssembler(inputCols=assembler_cols, outputCol="features")
stages += [assembler]


# In[13]:


cols


# In[14]:


# 使用pipeline完成数据处理
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(df)
df = pipeline_model.transform(df)
selected_cols = ["label", "features"] + cols
df = df.select(selected_cols)
df.show(3)


# In[15]:


import pandas as pd
pd.DataFrame(df.take(20), columns = df.columns)


# In[16]:


display(df)


# In[17]:


# 数据集切分
train, test = df.randomSplit([0.7, 0.3], seed=2021)
print(train.count())
print(test.count())


# ### Logistic Regression

# In[18]:


from pyspark.ml.classification import LogisticRegression
# 创建模型
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label',maxIter=10)
lr_model = lr.fit(train)


# In[19]:


predictions = lr_model.transform(test)
predictions.take(1)


# In[20]:


predictions.printSchema()


# In[21]:


selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)
selected.show(4)


# In[22]:


# rawPrediction 预测的原始数据，prediction分类结果
pd.DataFrame(predictions.take(4), columns = predictions.columns)


# In[23]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
# 模型评估，通过原始数据 rawPrediction计算AUC
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('AUC：', evaluator.evaluate(predictions))


# In[24]:


evaluator.getMetricName()


# In[25]:


print(lr.explainParams())


# In[26]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 创建网络参数，用于交叉验证
param_grid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())


# In[27]:


# 五折交叉验证，设置模型，网格参数，验证方法，折数
cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
# 交叉验证运行
cv_model = cv.fit(train)


# In[28]:


# 查看cv有哪些参数可以调整
print(cv.explainParams())


# In[29]:


# 对于测试数据，使用五折交叉验证
predictions = cv_model.transform(test)
print('AUC：', evaluator.evaluate(predictions))


# ### Decision Trees

# In[30]:


from pyspark.ml.classification import DecisionTreeClassifier

# 创建决策树模型
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dt_model = dt.fit(train)


# In[31]:


print(dt_model._call_java('toDebugString'))


# In[32]:


print("numNodes = ", dt_model.numNodes)
print("depth = ", dt_model.depth)


# In[33]:


# 模型预测
predictions = dt_model.transform(test)
predictions.printSchema()


# In[34]:


selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)


# In[35]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)


# In[36]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
param_grid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 6, 10])
             .addGrid(dt.maxBins, [20, 40, 80])
             .build())


# In[37]:


# 设置五折交叉验证
cv = CrossValidator(estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
# 运行cv
cv_model = cv.fit(train)


# In[38]:


# 查看最优模型
print("numNodes = ", cv_model.bestModel.numNodes)
print("depth = ", cv_model.bestModel.depth)


# In[39]:


# 使用五折交叉验证进行预测
predictions = cv_model.transform(test)
evaluator.evaluate(predictions)


# In[40]:


selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)


# In[41]:


selected.show(3)


# ### Random Forest

# In[42]:


from pyspark.ml.classification import RandomForestClassifier
# 随机森林
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rf_model = rf.fit(train)
predictions = rf_model.transform(test)
predictions.printSchema()


# In[43]:


selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)


# In[44]:


selected.show(3)


# In[45]:


evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)


# In[46]:


# RF调参
param_grid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [3, 5, 7])
             .addGrid(rf.maxBins, [20, 50])
             .addGrid(rf.numTrees, [5, 10])
             .build())

# 五折交叉验证
cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
# 运行CV（大约6分钟）
cv_model = cv.fit(train)


# In[47]:


predictions = cv_model.transform(test)
evaluator.evaluate(predictions)


# In[48]:


selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)


# In[49]:


selected.show(3)


# ### 使用RF best_model

# In[50]:


best_model = cv_model.bestModel
final_predictions = best_model.transform(df)
evaluator.evaluate(final_predictions)


# In[51]:


best_model


# In[52]:


# 查看特征重要性
best_model.featureImportances


# In[55]:


# 查看featuers的metadata
df.schema['features'].metadata


# In[68]:


temp = df.schema["features"].metadata["ml_attr"]["attrs"]
temp


# In[ ]:


# 添加 numeric feature importance
df_importance = pd.DataFrame(columns=['idx', 'name'])
for attr in temp['numeric']:
    temp_df = {}
    temp_df['idx'] = attr['idx']
    temp_df['name'] = attr['name']
    #print(temp_df)
    df_importance = df_importance.append(temp_df, ignore_index=True)
    #print(attr['idx'], attr['name'])
    #print(attr)
    #break
df_importance


# In[77]:


# 添加 binary feature importance
for attr in temp['binary']:
    temp_df = {}
    temp_df['idx'] = attr['idx']
    temp_df['name'] = attr['name']
    df_importance = df_importance.append(temp_df, ignore_index=True)
df_importance


# In[87]:


print(best_model.featureImportances)
df_temp = pd.DataFrame(best_model.featureImportances.toArray())
df_temp.columns = ['feature_importance']
df_importance = df_importance.merge(df_temp, left_index=True, right_index=True)
df_importance


# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(40, 30))
# 不设置order参数的结果（默认情况下）
sns.barplot(x="feature_importance", y="name", data=df_importance, orient="h")


# In[93]:


df_importance.sort_values(by=['feature_importance'], ascending=False, inplace=True)
df_importance


# In[94]:


plt.figure(figsize=(40, 30))
# 不设置order参数的结果（默认情况下）
sns.barplot(x="feature_importance", y="name", data=df_importance, orient="h")

