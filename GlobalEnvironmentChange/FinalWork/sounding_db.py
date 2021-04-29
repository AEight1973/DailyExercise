import pandas as pd
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database


def csv2db(_path):
    _, dbname, tablename = _path.split('/')
    con_engine = create_engine(
        'mysql+pymysql://root:GISChaser521_p@ssw0rd@localhost:3306/{}'.format('sounding' + dbname))

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

    data = pd.read_csv(_path)
    metadata.create_all(con_engine)

    for i in range(len(data)):
        session.add(Record(pressure=data.iloc[i, 0],
                           height=data.iloc[i, 1],
                           temperature=data.iloc[i, 2],
                           dewpoint=data.iloc[i, 3],
                           direction=data.iloc[i, 4],
                           speed=data.iloc[i, 5],
                           u_wind=data.iloc[i, 6],
                           v_wind=data.iloc[i, 7]))
        session.commit()
