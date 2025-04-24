# Importing all the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Data Processing

titanic_df = pd.read_csv("Titanic-Dataset.csv") # Creating Titanic DataFrame

print(titanic_df.head()) # Prints first 5 rows from the Titanic Dataset

print(titanic_df.isnull().sum()) # Gives the number of missing values.

# drop the "Cabin" column from the dataframe
titanic_df = titanic_df.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

# finding the mode value of "Embarked" column
print(titanic_df['Embarked'].mode())

print(titanic_df['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# check the number of missing values in each column
titanic_df.isnull().sum()

# getting some statistical measures about the data
titanic_df.describe()

# finding the number of people survived and not survived
titanic_df['Survived'].value_counts()

# Data Visualization

sns.set()

# making a count plot for "Survived" column
sns.countplot('Survived', data=titanic_df)

titanic_df['Sex'].value_counts()

# making a count plot for "Sex" column
sns.countplot('Sex', data=titanic_df)

# number of survivors Gender wise
sns.countplot('Sex', hue='Survived', data=titanic_df)

# making a count plot for "Pclass" column
sns.countplot('Pclass', data=titanic_df)

sns.countplot('Pclass', hue='Survived', data=titanic_df)

# Encoding the Categorical Columns

titanic_df['Sex'].value_counts()

titanic_df['Embarked'].value_counts()

# converting categorical Columns

titanic_df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

titanic_df.head()

# Separating features & Target

X = titanic_df.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_df['Survived']

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2) 

print(X.shape, X_train.shape, X_test.shape)

# Model Training

# Logistic Regression

model = LogisticRegression()

# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Model Evaluation

# Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)

print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)

print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)