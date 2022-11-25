#importing Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")


print(data.head(10))

print(data.columns)

new_data1=data.dropna()

X = new_data1[["Store ID","Total Price","Base Price"]]
Y = new_data1[["Units Sold"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
model = DecisionTreeRegressor()
model.fit(X,Y)

pickle.dump(model,open("model.pkl","wb"))