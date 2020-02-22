import pandas as pd
from Logistic_regression import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


class Predictor():
    def __init__(self, file_name, target_column, source_column):
        self.file_name = file_name
        self.data_columns = source_column
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def run_model(self, learning_rate, n_iterations):
        df = pd.read_csv(self.file_name, sep=',')
        y = df[self.target_column].to_numpy()

        X = df[self.data_columns].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        regressor = LogisticRegression(learning_rate, n_iterations)
        regressor.fit_model(self.X_train, self.y_train)
        predictions = regressor.make_prediction(self.X_test)
        return predictions

    def get_performance(self, predictions):
        print("LR classification accuracy:", self.accuracy(self.y_test, predictions))

    def confusion_matrix(self,predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(predictions)):
            if self.y_test[i] == predictions[i] == 1:
                TP += 1
            if predictions[i] == 1 and self.y_test[i] != predictions[i]:
                FP += 1
            if self.y_test[i] == predictions[i] == 0:
                TN += 1
            if predictions[i] == 0 and self.y_test[i] != predictions[i]:
                FN += 1
        TPR = TP / (TP + FN)
        FPR = FP/(FP+TN)
        tpr=[]
        tpr.append(0.0)
        tpr.append(TPR)
        tpr.append(1.0)
        fpr = []
        fpr.append(0.0)
        fpr.append(FPR)
        fpr.append(1.0)

        print(tpr,'mytpr')
        print(fpr,'myfpr')
        plt.show()
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, predictions, pos_label=0)
        print(fpr,'curve-fpr')
        print(tpr,'curve-tpr')
        #Print ROC curve
        # plt.plot(fpr, tpr)
        # plt.show()

        # This is the AUC


    def roc(self,predicted_probabilities,predictions):
        w=2
        h=2

        fpr = []
        # true positive rate
        tpr = []
        # Iterate thresholds from 0.0, 0.01, ... 1.0
        thresholds = np.arange(0.0, 1.01, .1)
        P = sum(self.y_test)
        N = len(self.y_test) - P
        for thresh in thresholds:
            matrix = [[0 for x in range(w)] for y in range(h)]
            FP = 0
            TP = 0
            TN=0
            FN=0
            for i in range(len(predicted_probabilities)):
                if predicted_probabilities[i] > thresh:
                    if self.y_test[i] == 1:
                        TP = TP + 1
                    if self.y_test[i] == 0:
                        FP = FP + 1
                elif predicted_probabilities[i]<thresh:
                    if self.y_test[i]== 0:
                        TN += 1
                    if self.y_test[i] == 1:
                        FN += 1
            fpr.append(FP / float(N))
            tpr.append(TP / float(P))
            matrix[0][0]=TP
            matrix[0][1]=FP
            matrix[1][0]=FN
            matrix[1][1]=TN


        plt.plot(fpr, tpr)
        auc = roc_auc_score(self.y_test, predictions)
        print(auc,'sfasdf')
        plt.show()
        return


if __name__ == '__main__':
    b = run('titanic.csv', 'Survived', ['Pclass', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare'])
    predictions,predicted_probabilities = b.run_model(learning_rate=0.0001, n_iterations=1000)
    b.get_performance(predictions)
    #b.confusion_matrix(predictions)
    b.roc(predicted_probabilities,predictions)
