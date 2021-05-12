# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import DBSCAN
from flask import Flask, request, jsonify, render_template

dataFrame = pd.read_json("C:\\Users\\abhis\\Downloads\\MOCK_DATA.json")
dataFrame.head()

app = Flask(__name__)
safe_distance = 0.0018288 # a radial distance of 6 feet in kilometers
model = DBSCAN(eps=safe_distance, min_samples=2, metric='haversine').fit(dataFrame[['Latitude', 'Longitude']])

core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True
labels = model.labels_
dataFrame['Cluster'] = model.labels_.tolist()


@app.route('/')
def home():
    title = 'covid'
    return render_template('index.html',title=title)

@app.route('/predict',methods=['POST'])
def predict():
    inputName = request.form.get("name")
    #Check if name is valid
    assert (inputName in dataFrame['User'].tolist()), print("User Doesn't exist")
    #Social distance
    safe_distance = 0.0018288 #6 feets in kilometers
    #Apply model, in case of larger dataset or noisy one, increase min_samples
    model = DBSCAN(eps=safe_distance, min_samples=2, metric='haversine').fit(dataFrame[['Latitude', 'Longitude']])
    #Get clusters found bt the algorithm 
    labels = model.labels_
    #Add the clusters to the dataframe
    dataFrame['Cluster'] = model.labels_.tolist()
    #Get the clusters the inputName is a part of
    inputNameClusters = set()
    for i in range(len(dataFrame)):
        if dataFrame['User'][i] == inputName:
            inputNameClusters.add(dataFrame['Cluster'][i])
    #Get people who are in the same cluster as the inputName              
    infected = set()
    for cluster in inputNameClusters:
        if cluster != -1: #as long as it is not the -1 cluster
            namesInCluster = dataFrame.loc[dataFrame['Cluster'] == cluster, 'User'] #Get all names in the cluster
            for i in range(len(namesInCluster)):
                #locate each name on the cluster
                name = namesInCluster.iloc[i]
                if name != inputName: #Don't want to add the input to the results
                    infected.add(name)
    #print("Potential infections are:",*infected,sep="\n" )
    

    output = ', '.join(infected)

    
    return render_template('result.html', prediction=output)



if __name__ == "__main__":
    app.run(debug=True)
    