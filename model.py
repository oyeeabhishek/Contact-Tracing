import pandas as pd
import numpy as np
import pickle
import datetime as dt
from sklearn.cluster import DBSCAN

dataFrame = pd.read_json("C:\\Users\\abhis\\Downloads\\MOCK_DATA.json")
dataFrame.head()

safe_distance = 0.0018288 # a radial distance of 6 feet in kilometers
model = DBSCAN(eps=safe_distance, min_samples=2, metric='haversine').fit(dataFrame[['Latitude', 'Longitude']])
core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True
labels = model.labels_
dataFrame['Cluster'] = model.labels_.tolist()

pickle.dump(model, open('modelp.pkl','wb'))

modelp = pickle.load(open('modelp.pkl','rb'))
