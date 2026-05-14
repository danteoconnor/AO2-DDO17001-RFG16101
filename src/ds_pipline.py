from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #added for plotting

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
#store the dataset in a DataFrame
df = housing.frame
print(df.head())

#Split the dataset into Training, Validation, and Test sets
from sklearn.model_selection import train_test_split

#Define target variable and features
target = 'MedHouseVal'
features = df.drop([target], axis=1)
target_variable = df[target]

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target_variable, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#Print the shapes of the datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)
