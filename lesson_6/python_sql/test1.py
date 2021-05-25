import mysql.connector
# 打开数据库连接
db = mysql.connector.connect(
       host="localhost",
       user="root",
       passwd="passw0rdcc4", # 写上你的数据库密码
       database='wucai', 
       auth_plugin='mysql_native_password'
)
# 获取操作游标 
cursor = db.cursor()
# 执行SQL语句
cursor.execute("SELECT VERSION()")
# 获取一条数据
data = cursor.fetchone()
print("MySQL版本: %s " % data)
# 关闭游标&数据库连接
cursor.close()
db.close()
