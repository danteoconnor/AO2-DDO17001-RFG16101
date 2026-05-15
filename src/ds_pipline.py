from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split #added for plotting
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
#importing necessary libraries for data manipulation, visualization, and numerical operations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train MLPRegressor with custom hyperparameters and early stopping
mlp = MLPRegressor(
    alpha=0.0001,                   # Custom hyperparameter Alpha
    learning_rate_init=0.001,       # Custom hyperparameter initial learning rate
    early_stopping=True,            # Enable early stopping
    activation='relu',              # Custom hyperparameter activation function
    random_state=42
)

#Fit the model to the training data
mlp.fit(X_train_scaled, y_train)

# Evaluate the model on the validation set
val_score = mlp.score(X_val_scaled, y_val)
print("Validation R^2 Score:", val_score)

# Evaluate the model on the test set
test_score = mlp.score(X_test_scaled, y_test)
print("Test R^2 Score:", test_score)

#both of the above our scores give a result over .74 which means that our fitted model is able to represent over 74% of the data variance. 
# This is a good result for a regression model, indicating that it has learned to capture the underlying patterns in the data effectively. 

#Predictions for train, validation, and test sets
y_train_pred = mlp.predict(X_train_scaled)
y_val_pred = mlp.predict(X_val_scaled)
y_test_pred = mlp.predict(X_test_scaled)

#Print the first 5 predictions for each set
print("Predictions for Training Set:", y_train_pred[:5])
print("Predictions for Validation Set:", y_val_pred[:5])
print("Predictions for Test Set:", y_test_pred[:5]) 

#Create scatterplot of predicted vs actual for Training data
plt.figure(figsize=(6,6))
plt.scatter(y_train, y_train_pred, alpha=0.3, s=10)
#Plot a reference line
lo = min(np.min(y_train), np.min(y_train_pred))
hi = max(np.max(y_train), np.max(y_train_pred))
plt.plot([lo, hi], [lo, hi], linewidth=1, color='red') 
#Set Axis ranges
plt.xlim(0,6)
plt.ylim(0,6)
#Set labels and title
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Predicted vs Actual — Train")
plt.tight_layout()
plt.savefig("figures/train_actual_vs_pred.png")
plt.show()
plt.close()

#Create scatterplot of predicted vs actual for Validation data
plt.figure(figsize=(6,6))
plt.scatter(y_val, y_val_pred, alpha=0.3, s=10)
#Plot a reference line
lo = min(np.min(y_val), np.min(y_val_pred))
hi = max(np.max(y_val), np.max(y_val_pred))
plt.plot([lo, hi], [lo, hi], linewidth=1, color='red') 
#Set Axis ranges
plt.xlim(0,6)
plt.ylim(0,6)
#Set labels and title
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Predicted vs Actual — Validation")
plt.tight_layout()
plt.savefig("figures/validation_actual_vs_pred.png")
plt.show()
plt.close()

#Create scatterplot of predicted vs actual for Test data
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.3, s=10)
#Plot a reference line
lo = min(np.min(y_test), np.min(y_test_pred))
hi = max(np.max(y_test), np.max(y_test_pred))
plt.plot([lo, hi], [lo, hi], linewidth=1, color='red') 
#Set Axis ranges
plt.xlim(0,6)
plt.ylim(0,6)
#Set labels and title
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Predicted vs Actual — Test")
plt.tight_layout()
plt.savefig("figures/test_actual_vs_pred.png")
plt.show()
plt.close()
