# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# For ignoring warning
import warnings
warnings.filterwarnings("ignore")

# Read and Load dataset
df = pd.read_csv("C:/Users/user/Downloads/survey lung cancer.csv")
df.columns = ['Gender', 'Age', 'Smoking', 'Yellow_Fingers', 'Anxiety', 
      'Peer_Pressure', 'Chronic_Disease', 'Fatigue', 'Allergy', 'Wheezing', 
      'Alcohol_Consuming', 'Coughing', 'Shortness_of_Breath', 'Swallowing_Difficulty', 
      'Chest_Pain', 'Lung_Cancer']
df

# Exploratory Data Analysis
df.info()
df.head(5)
df.tail(5)
df.shape

df[df['Lung_Cancer']=='YES'].count()

# Check Missing Values
df.isnull().sum()

# Check Duplicated data
df.duplicated().value_counts()

# Remove Duplicated data
df.drop_duplicates(inplace=True)

df.shape

df.describe()

# สถิติเบื้องต้นของ target variable
df[df['Lung_Cancer'] == 'YES'].count()
df['Lung_Cancer'].value_counts(normalize=True)

# Check the Distibution
# Check the Distribution for Target variable ('Lung_Cancer')
sns.countplot(x='Lung_Cancer', data=df, color='orange')
plt.show()

df['Lung_Cancer'].value_counts()

# Check the distribution between independent variables and target variable
def plot(col, df=df):
    return df.groupby(col)['Lung_Cancer'].value_counts(normalize=True).unstack().plot(kind='bar'
           , figsize=(8,5))
  
plot('Age')
plot('Smoking')
plot('Yellow_Fingers')
plot('Allergy')
plot('Wheezing')
plot('Alcohol_Consuming')
plot('Coughing')
plot('Shortness_of_Breath')
plot('Swallowing_Difficulty')
plot('Chest_Pain')

# Find Correlation
corr
df1 = df.drop(columns=['Gender','Age','Smoking','Shortness_of_Breath'], axis=1)
df1

cmap=sns.light_palette("green", as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(corr,cmap=cmap,annot=True, square=True)
plt.show()

# Convert columns to numerical values using LabelEncoder from sklearn
le = LabelEncoder()
df1['Yellow_Fingers'] = le.fit_transform(df1['Yellow_Fingers'])
df1['Anxiety'] = le.fit_transform(df1['Anxiety'])
df1['Peer_Pressure'] = le.fit_transform(df1['Peer_Pressure'])
df1['Chronic_Disease'] = le.fit_transform(df1['Chronic_Disease'])
df1['Fatigue'] = le.fit_transform(df1['Fatigue'])
df1['Allergy'] = le.fit_transform(df1['Allergy'])
df1['Wheezing'] = le.fit_transform(df1['Wheezing'])
df1['Alcohol_Consuming'] = le.fit_transform(df1['Alcohol_Consuming'])
df1['Coughing'] = le.fit_transform(df1['Coughing'])
df1['Swallowing_Difficulty'] = le.fit_transform(df1['Swallowing_Difficulty'])
df1['Chest_Pain'] = le.fit_transform(df1['Chest_Pain'])
df1['Lung_Cancer'] = le.fit_transform(df1['Lung_Cancer'])

df1
df1.info()

# Split the dataset into the Training set and Test set
## Feature Engineering
kot = corr[corr>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Blues")

df1['Scale'] = df1['Anxiety']*df1['Yellow_Fingers']

# Drop Lung_Cancer column 
X = df1.drop(columns=['Lung_Cancer'], axis=1)
y = df1['Lung_Cancer']

# Use Method ADASYN for handling Imbalanced Data
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

len(X_resampled)

# Split Traning set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, 
                      y_resampled, test_size=0.2, random_state=42)

# K-Nearest Neighbors
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the KNN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_knn_pred = classifier.predict(X_test)
print(y_knn_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_knn_pred)
print(f'Accuracy: {accuracy}')

cm = confusion_matrix(y_test, y_knn_pred)
print(cm)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(y_test, y_knn_pred)
print(class_report)

###########################################################################################

# Artificial Neural Network
from sklearn.neural_network import MLPClassifier
# Create an MLPClassier model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
# Train the model
mlp.fit(X_train, y_train)

# Predict X_test
y_ann_pred = mlp.predict(X_test)
print(y_ann_pred)

# Accuracy score and Confusion Matrix
accuracy = accuracy_score(y_test, y_ann_pred)
print(f'Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test, y_ann_pred)
print(conf_matrix)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(y_test, y_ann_pred)
print('Classification Report:')
print(class_report)




#!/usr/bin/env python




