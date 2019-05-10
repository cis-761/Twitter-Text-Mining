
import numpy as np  # Data manipulation
import decimal
import pandas as pd           # Dataframe manipulatio
import geopandas
from shapely.geometry import Point
import matplotlib.pyplot as plt
import conda
import os
import math
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library\\share'))
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
from scipy.cluster.vq import kmeans2, whiten


tweet_read =pd.read_csv('tweets.csv', delimiter=',' , )
#get not null location tweets
tweet_withlocation = tweet_read[tweet_read.location.notnull()]
#important_text= ['flu', 'flushot', 'sick', 'ill', 'cold', 'influenza']
tweet_final =tweet_withlocation[tweet_withlocation['text'].str.contains('flu|flushot|sick|ill|cold|influenza', regex = True)]
tweet=tweet_final['location'].str.strip('[]')                               \
                   .str.split(', ', expand=True)                   \
                   .rename(columns={0:'lon', 1:'lat'})
tweet['geometry'] = tweet.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
tweet['lat'] = tweet.apply(lambda x : float(x.lat), axis =1)
tweet['lon'] = tweet.apply(lambda x : float(x.lon), axis =1)
lat = tweet['lat'].tolist()
lon = tweet['lon'].tolist()
# margin = 2 # buffer to add to the range
# lat_min = min(lat) - margin
# lat_max = max(lat) + margin
# lon_min = min(lon) - margin
# lon_max = max(lon) + margin
# print(lat_min)
# lat_min2=int(round(lat_min))
# lat_max2 = int(round(lat_max))
# lon_min2 = int(round(lon_min))
# lon_max2 = int(round(lon_max))
# lat_avg = int((lat_max-lat_min)/2)
# lon_avg = int((lon_max-lon_min)/2)
# m = Basemap(llcrnrlon=lat_min2,
#             llcrnrlat=lon_min2,
#             urcrnrlon=lon_max2,
#             urcrnrlat=lat_max2,
#             lat_0=lat_avg,
#             lon_0=lon_avg,
#             projection='merc',
#             resolution = 'i'
#             )
m = Basemap(projection='merc',
            llcrnrlat = -35,
            llcrnrlon = -170,
            urcrnrlat = 60,
            urcrnrlon = 170,
            lat_0=50,
            lon_0=165,
            resolution='h')
m.drawcoastlines()
m.drawcountries(linewidth=2)
m.drawstates(color= 'b')
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
m.bluemarble()
lons, lats = m(lon, lat)
#plot points as red dots
m.scatter(lons, lats, marker = 'o', color='r', zorder=5)
plt.show()

