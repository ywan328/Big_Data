pip install sqlalchemy -i https://pypi.tuna.tsinghua.edu.cn/simple

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

import traceback
try:
  sql = "INSERT INTO player (team_id, player_name, height) VALUES (%s, %s, %s)"
  val = (1003, "约翰-科林斯", 2.08)
  cursor.execute(sql, val)
  db.commit()
  print(cursor.rowcount, "记录插入成功。")
except Exception as e:
  # 打印异常信息
  traceback.print_exc()
  # 回滚  
  db.rollback()
finally:
  # 关闭数据库连接
  db.close()

