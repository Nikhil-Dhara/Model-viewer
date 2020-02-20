
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
#import matplotlib.pyplot as plt
from Logistic_regression import LogisticRegression


# from regression import LogisticRegression

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy




df=pd.read_csv('titanic.csv', sep=',')
y=df['Survived'].to_numpy()
X=df[['Pclass','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']].to_numpy()
bc = datasets.load_breast_cancer()
X1, y1 = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
regressor = LogisticRegression(learning_rate=0.0001, n_iterations=1000)
regressor.fit_model(X_train, y_train)
predictions = regressor.make_prediction(X_test)

print("LR classification accuracy:", accuracy(y_test, predictions))