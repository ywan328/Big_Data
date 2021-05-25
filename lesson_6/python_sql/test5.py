import sqlite3

# 使用SQLite3创建数据库连接
conn = sqlite3.connect("wucai.db")
# 如果没有这个文件存储，会自动进行创建，然后可以使用conn操作连接，通过会话连接conn来创建游标
cur = conn.cursor()
# 使用execute()方法来执行各种DML，比如插入，删除，更新等，也可以进行SQL查询
cur.execute("CREATE TABLE IF NOT EXISTS heros (id int primary key, name text, hp_max real, mp_max real, role_main text)")
# 使用execute()方法来添加一条数据
#cur.execute('insert into heros values(?, ?, ?, ?, ?)', (10000, '夏侯惇', 7350, 1746, '坦克'))
# 也可以批量插入，这里会使用到executemany方法，传入的参数就是一个元组
cur.executemany('insert into heros values(?, ?, ?, ?, ?)', 
           ((10000, '夏侯惇', 7350, 1746, '坦克'),
            (10001, '钟无艳', 7000, 1760, '战士'),
          (10002, '张飞', 8341, 100, '坦克')))
# 想要对heros数据表进行查询，同样使用execute执行SQL语句
cur.execute("SELECT id, name, hp_max, mp_max, role_main FROM heros")
# 提交事务 
conn.commit()
# 关闭游标
cur.close()
# 关闭数据库连接
conn.close()


