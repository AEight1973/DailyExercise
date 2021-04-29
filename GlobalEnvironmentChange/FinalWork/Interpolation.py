import pyKriging as krige


'''
空间插值
'''
def space_kriging(train_x, train_y):
    model = krige.kriging(train_x, train_y, name='simple')
    model.train(optimizer='ga')
    # model.predict([x, y])
    model.plot()


'''
时间插值
'''
def time_gru(train_x, train_y):
    return
