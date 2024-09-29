import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


agriculture_data=pd.read_csv('/content/Crop_recommendation.csv')
agriculture_data.shape()
agriculture_data.head()
agriculture_data.duplicated().sum()
agriculture_data.describe()
sns.heatmap(agriculture_data.isnull())
numeric_data = agriculture_data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
sns.heatmap(correlation,annot=True)
#traing and testing split
X=agriculture_data.drop("label",axis=1)
Y=agriculture_data['label']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train

GaussianNB=GaussianNB()
GaussianNB.fit(X_train,Y_train)

# Ensure the directory exists
directory = 'C:/Users/user/Desktop/AGRICULTURE_PRED'
os.makedirs(directory, exist_ok=True)

# Construct the absolute file path
file_path = os.path.join(directory, 'result.pkl')

# Example data to be pickled
data = {'example': 'data'}

try:
    # Open the file in write-binary mode
    with open(file_path, 'wb') as f:
        # Pickle the data and write it to the file
        pickle.dump(data, f)
    print("Pickle file created successfully.")
except Exception as e:
    print("Error:", e)

pickle.dump(GaussianNB,open("result.pkl","wb"))