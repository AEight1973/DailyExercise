import pyKriging as krige


def space_kriging(train_x, train_y):
    model = krige.kriging(train_x, train_y, name='simple')
    model.train(optimizer='ga')
    # model.predict([x, y])
    model.plot()


def time_kriging(train_x, train_y):
    return
