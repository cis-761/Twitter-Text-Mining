from scipy.cluster.vq import kmeans2, whiten
import numpy as np  # Data manipulation
import decimal
import pandas as pd           # Dataframe manipulatio
import geopandas
from shapely.geometry import Point
import matplotlib.pyplot as plt



# #print(whiten(coordinates))
# coordinates = np.column_stack((lon, lat))
# #print(coordinates)
# x, y = kmeans2(whiten(coordinates), 3, iter = 20)
# print(x, y)
# plt.scatter(coordinates[:,0], coordinates[:,1], c=y);
# plt.show()