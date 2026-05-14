from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt #added for plotting

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)