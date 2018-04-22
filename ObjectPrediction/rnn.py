import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def hashCoords(x, y):
	return (y*1000 + x)/(608608)	

dataset = np.array([[]])

#hash for 608 by 608 image coordinate: (y*1000 + x)/(608608)