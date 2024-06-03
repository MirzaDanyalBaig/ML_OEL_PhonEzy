import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('ML-OEL\smartphonesDataset.csv')
print(df.head()) #initial 5 rows of the dataset

#Data Preprocessing
print(df.isnull().sum()) #checking for missing values
#Missing values are present in the dataset in RAM and Storage columns

print(df.describe()) #summary of the dataset

print(df.info()) #information about the dataset

print(df['Brand'].value_counts()) #count of each brand in the dataset
print(df['RAM'].value_counts()) #count of each RAM in the dataset
print(df['Storage'].value_counts()) #count of each Storage in the dataset

print(df.duplicated().sum()) #checking for duplicate values