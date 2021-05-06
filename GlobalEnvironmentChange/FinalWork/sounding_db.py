import os
import pandas as pd
from sqlalchemy import Column, Integer, create_engine, SmallInteger, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
from MoveFile import movefile


def csv2db(_path, session=None):
    _, dbname, tablename = _path.split('/')
    con_engine = create_engine(
        'mysql+pymysql://root:AEight19731224@localhost:3306/{}'.format('sounding_' + dbname))

    # 若不存在，则新建数据库
    if not database_exists(con_engine.url):
        create_database(con_engine.url)

    Base = declarative_base()
    metadata = Base.metadata
    DBSession = sessionmaker(bind=con_engine)
    session = DBSession()


    class Record(Base):
        __tablename__ = 'record_' + tablename.split('_')[1][:-4]

        id = Column(SmallInteger, primary_key=True, autoincrement=True)
        height = Column(Integer)
        pressure = Column(Float)
        temperature = Column(Float)
        dewpoint = Column(Float)
        direction = Column(Float)
        speed = Column(Float)


    data = pd.read_csv(_path)
    data_nan = data.notnull()
    metadata.create_all(con_engine)

    for i in range(len(data)):
        session.add(Record(pressure=data.iloc[i, 0] if data_nan.iloc[i, 0] else None,
                           height=data.iloc[i, 1] if data_nan.iloc[i, 1] else None,
                           temperature=data.iloc[i, 2] if data_nan.iloc[i, 2] else None,
                           dewpoint=data.iloc[i, 3] if data_nan.iloc[i, 3] else None,
                           direction=data.iloc[i, 4] if data_nan.iloc[i, 4] else None,
                           speed=data.iloc[i, 5] if data_nan.iloc[i, 5] else None))
    session.commit()
    session.close()

    movefile(_path, 'E:/DataBackup/sounding_station/' + dbname + '/' + tablename)


def read(_station, _record):
    con_engine = create_engine(
        'mysql+pymysql://root:AEight19731224@localhost:3306/{}'.format('sounding_' + _station))

    Base = declarative_base()
    metadata = Base.metadata
    DBSession = sessionmaker(bind=con_engine)
    session = DBSession()

    class Record(Base):
        __tablename__ = 'record_' + _record

        id = Column(SmallInteger, primary_key=True, autoincrement=True)
        height = Column(Integer)
        pressure = Column(Float)
        temperature = Column(Float)
        dewpoint = Column(Float)
        direction = Column(Float)
        speed = Column(Float)

        def to_dict(self):
            return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    return [ven.to_dict() for ven in session.query(Record).all()]


if __name__ == '__main__':
    from datetime import datetime
    stationlist = os.listdir('cache/data')
    for station in stationlist[167:]:
        print('{0} 开始写入{1}数据库'.format(datetime.now().isoformat(), 'sounding_'+station))
        recordlist = os.listdir('data/' + station)
        if os.path.exists('cache/data/' + station  + '/download.json'):
            recordlist.remove('download.json')
        for record in recordlist:
            csv2db('cache/data/' + station + '/' + record)
            print('--> {0} 成功写入表{1}'.format(datetime.now().isoformat(), 'record_' + record[:-4]))
        print('{0} 成功写入{1}数据库'.format(datetime.now().isoformat(), 'sounding_' + station))
