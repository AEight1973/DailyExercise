from sqlalchemy import Column, Float, Integer, Text, create_engine
from sqlalchemy.dialects.mysql import TINYTEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
import pandas as pd


def csv2db(_path):
    _, dbname, tablename = _path.split('/')
    con_engine = create_engine('mysql+pymysql://root:GISChaser521_p@ssw0rd@localhost:3306/{}'.format('sounding' + dbname))

    # 若不存在，则新建数据库
    if not database_exists(con_engine.url):
        create_database(con_engine.url)

    Base = declarative_base()
    metadata = Base.metadata
    DBSession = sessionmaker(bind=con_engine)
    session = DBSession()

    class Record(Base):
        __tablename__ = 'Record' + tablename

        id = Column(Integer, primary_key=True, autoincrement=True)
        height = Column(Integer)
        pressure = Column(Integer)
        temperature = Column(Integer)
        dewpoint = Column(Integer)
        direction = Column(Integer)
        speed = Column(Integer)
        u_wind = Column(Integer)
        v_wind = Column(Integer)

        def to_dict(self):
            return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def create(_name, _content, _tag):
    return Comments(name=_name,
                    content=_content,
                    tag=_tag)


def get(_name=None):
    # 创建Query查询，filter是where条件，最后调用one()返回唯一行，
    if _name is None:
        _data = session.query(Comments).all()
    else:
        _data = session.query(Comments).filter_by(name=_name).all()
    return _data


def insert(_name, _content, _req):
    metadata.create_all(con_engine)
    session.add(create(_name, _content, ReqToText(_req)))
    session.commit()
    return {'code': '201000', 'message': '入库成功'}


if __name__ == '__main__':
    with open("../data/backup/comment.txt", 'r') as f:
        data = eval(f.read())
    metadata.create_all(con_engine)
    for i in data:
        session.add(create(i['name'], i['content'], i['tag']))
        session.commit()
