import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('smartphonesDataset.csv')
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

# Cleaning of data
# Filling ram and storage with mode of there respective columns 
df.fillna({'RAM': df['RAM'].mode()[0]} , inplace = True)
df.fillna({'Storage': df['Storage'].mode()[0]} , inplace = True)

# EDA
# Visualizing brand distribution
'''count = df["Brand"].value_counts()
# Calculate percentages
percentage = (count/ count.sum()) * 100

# Create the horizontal bar plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=percentage, y=percentage.index, orient='h',palette="deep")

# Annotate each bar with the percentage value
for index, value in enumerate(percentage):
    plt.text(value, index, f'{value:.2f}%', va='center')

# Set the labels and title
plt.xlabel('Percentage')
plt.ylabel("Brand")
plt.title('Brand Distribution')
plt.show()

# Visualizing RAM distribution
# Count the occurrences of each RAM category
count = df["RAM"].value_counts()

# Calculate percentages
percentage = (count / count.sum()) * 100

# Create a bar plot (histogram) with percentage labels
plt.figure(figsize=(10, 6))
barplot=sns.barplot(x=percentage.index, y=percentage, palette='muted')
# Annotate each bar with the percentage value
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.2f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 10),
                     textcoords='offset points')

# Set the labels and title
plt.title("RAM Distribution")
plt.xlabel("RAM")
plt.ylabel("Percentage")
plt.show()

# Visualize smartphone storage distribution
count=df["Storage"].value_counts()
# Calculate percentages
percentage = (count / count.sum()) * 100

# Create a bar plot (histogram) with percentage labels
plt.figure(figsize=(10, 6))
barplot=sns.barplot(x=percentage.index, y=percentage, palette='muted')
# Annotate each bar with the percentage value
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.2f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 10),
                     textcoords='offset points')
# Set the labels and title
plt.title("Storage Distribution")
plt.xlabel("Storage")
plt.ylabel("Percentage")
plt.show()

# Visualize color distribution
count = df["Color"].value_counts()
# Calculate percentages
percentage = (count/ count.sum()) * 100

# Create the horizontal barplot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=percentage, y=percentage.index, orient='h',palette="deep")
# Annotate each bar with the percentage value
for index, value in enumerate(percentage):
    plt.text(value, index, f'{value:.2f}%', va='center')
# Set the labels and title
plt.xlabel('Percentage')
plt.ylabel("Color")
plt.title('Color Distribution')
plt.show()

# Checking relation between RAM and storage
plt.scatter(x=df["RAM"],y=df["Storage"],c="green")
plt.title("the relation between RAM and Storage")
plt.xlabel("RAM")
plt.ylabel("Storage")
plt.show()

# Plot the distribution of each numeric feature
plt.figure(figsize=(20, 15))
df.hist(bins=30, figsize=(20, 15), color='teal', edgecolor='black')
plt.suptitle('Distribution of Numeric Features', fontsize=20)
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pairplot of Features', fontsize=20)
plt.show()
'''
# Feature Engineering: Creating new features
# Example: Price per GB of Storage (if 'Price' column exists)
if 'Price' in df.columns:
    df['Price_per_GB'] = df['Price'] / df['Storage']

# Assuming 'Price' and 'Price_per_GB' are numeric features
numeric_features = ['Price', 'Price_per_GB', 'RAM', 'Storage']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print(df.head())  # Check the new feature and scaled values
print(df['Price_per_GB'])