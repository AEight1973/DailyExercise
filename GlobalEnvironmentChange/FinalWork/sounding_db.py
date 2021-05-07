import json
import os
import pandas as pd
from sqlalchemy import Column, Integer, create_engine, SmallInteger, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
from MoveFile import movefile
import sqlalchemy
from RecordFailure import record_download


def csv2db(_path, _tag):
    _, _, dbname, tablename = _path.split('/')

    data = pd.read_csv(_path)
    result = examine(data, _tag, _path)
    if not result[0]:
        if not os.path.exists('E:/DataBackup/sounding_station/' + dbname + '/missing'):
            os.makedirs('E:/DataBackup/sounding_station/' + dbname + '/missing')
        movefile(_path, 'E:/DataBackup/sounding_station/' + dbname + '/missing/' + tablename)
        return _tag

    data_nan = data.notnull()

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

        pressure = Column(Float, primary_key=True)
        height = Column(Integer)
        temperature = Column(Float)
        dewpoint = Column(Float)
        direction = Column(Float)
        speed = Column(Float)

    metadata.create_all(con_engine)

    pre = 0
    for i in range(len(data)):
        if data.iloc[i, 0] == pre:
            continue
        else:
            pre = data.iloc[i, 0]
        session.add(Record(pressure=data.iloc[i, 0] if data_nan.iloc[i, 0] else None,
                           height=data.iloc[i, 1] if data_nan.iloc[i, 1] else None,
                           temperature=data.iloc[i, 2] if data_nan.iloc[i, 2] else None,
                           dewpoint=data.iloc[i, 3] if data_nan.iloc[i, 3] else None,
                           direction=data.iloc[i, 4] if data_nan.iloc[i, 4] else None,
                           speed=data.iloc[i, 5] if data_nan.iloc[i, 5] else None))
    session.commit()
    session.close()

    movefile(_path, 'E:/DataBackup/sounding_station/' + dbname + '/' + tablename)

    return result[1]


def read(_station, _record, _mode='simple', _return='dataframe'):
    con_engine = create_engine(
        'mysql+pymysql://root:AEight19731224@localhost:3306/{}'.format('sounding_' + str(_station)))

    Base = declarative_base()
    metadata = Base.metadata
    DBSession = sessionmaker(bind=con_engine)
    session = DBSession()

    class Record(Base):
        __tablename__ = 'record_' + str(_record)

        pressure = Column(Float, primary_key=True)
        height = Column(Integer)
        temperature = Column(Float)
        dewpoint = Column(Float)
        direction = Column(Float)
        speed = Column(Float)

        def to_dict(self):
            return {c.name: getattr(self, c.name) for c in self.__table__.columns if c.name != 'id'}

    if _mode == 'simple':
        output = []
        pressures = [850, 700, 500, 300, 200]
        for p in pressures:
            try:
                _data = session.query(Record).filter_by(pressure=p).one()
            except:
                continue
            output.append(_data.to_dict())
    else:
        output = [ven.to_dict() for ven in session.query(Record).all()]

    if len(output) == 0:
        return
    else:
        if _return == 'dataframe':
            return dict2pd(output)
        else:
            return output


def dict2pd(_data):
    return pd.DataFrame([list(i.values()) for i in _data], columns=list(_data[0].keys()))


def examine(_data, _tag, _path):
    p = _path.split('/')[-1].split('_')
    t = p[1][:-4]
    s = p[0]

    # 监测文件是否乱码，则删除并更改download文件
    try:
        _data.iloc[0, 1]
    except IndexError:
        record_download(s, t)
        os.remove(_path)
        return [False, _tag]

    # 监测文件是否出现缺失，则移动并跟更改download文件
    if _tag is None:
        return [True, float(_data.iloc[0, 1])]
    else:
        if _tag == float(_data.iloc[0, 1]):
            return [True, _tag]
        else:
            record_download(s, t, 'missing')
            return [False, _tag]


if __name__ == '__main__':
    from datetime import datetime

    stationlist = os.listdir('cache/data')
    for station in stationlist[167:]:
        print('{0} 开始写入{1}数据库'.format(datetime.now().isoformat(), 'sounding_' + station))
        recordlist = os.listdir('cache/data/' + station)
        if os.path.exists('cache/data/' + station + '/download.json'):
            recordlist.remove('download.json')
        tag = None
        for record in recordlist:
            tag = csv2db('cache/data/' + station + '/' + record, tag)
            print('--> {0} 成功写入表{1}'.format(datetime.now().isoformat(), 'record_' + record[:-4]))
        print('{0} 成功写入{1}数据库'.format(datetime.now().isoformat(), 'sounding_' + station))
