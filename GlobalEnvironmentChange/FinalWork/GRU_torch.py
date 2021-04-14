from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from math import sqrt
from LoadData import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
