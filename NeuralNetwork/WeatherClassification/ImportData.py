import cv2
import os
import numpy as np
from tqdm import tqdm

weather_type = {'cloudy': 0, 'haze': 1, 'rainy': 2, 'snow': 3, 'sunny': 4, 'thunder': 5}


def normalize(_url, _size):
    img = cv2.imread(_url, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (_size, _size))


def read():
    images = []
    label = []
    for _type, _code in weather_type.items():
        filename_list = os.listdir('data/WeatherClassification/' + _type + '/')
        print('<---------- 正在载入 {} ---------->'.format(_type))
        for _filename in tqdm(filename_list):
            _pathname = 'data/WeatherClassification/' + _type + '/' + _filename
            images.append(normalize(_pathname, 100))
            label.append(_code)
    return np.array(images), np.array(label)
