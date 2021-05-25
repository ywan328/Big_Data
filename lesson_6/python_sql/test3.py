from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.orm import sessionmaker

from sqlalchemy import Column, DateTime, Integer, Float, Index, String, Table, and_
from sqlalchemy.dialects.mysql import BIGINT, INTEGER, TINYINT, VARCHAR, DATETIME
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


# 创建&返回session
def get_db_session():
    engine = create_engine('mysql+mysqlconnector://root:passw0rdcc4@localhost:3306/wucai')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    session = DBSession()
    return engine, session
engine, session = get_db_session()

# 定义Player对象:
class Player(Base):
    # 表的名字:
    __tablename__ = 'player'
    # 表的结构:
    player_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer)
    player_name = Column(String(255))
    height = Column(Float(3,2))


# 创建session对象
DBSession = sessionmaker(bind=engine)
session = DBSession()
# 创建Player对象
new_player = Player(team_id = 1003, player_name = "约翰-科林斯", height = 2.08)
# 添加到session
session.add(new_player)
# 提交即保存到数据库
session.commit()
# 关闭session
session.close()


