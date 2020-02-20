import numpy as np


class LogisticRegression():
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = None
        self.weigths = None

    def fit_model(self, X, y):
        """

        :param X: NO of samples and features for each sample
        :param y: Target variable
        :return:
        """
        n_samples, n_features = X.shape
        self.bias = 0
        self.weigths = np.zeros(n_features)
        for _ in range(self.n_iterations):
            model = np.dot(X, self.weigths) + self.bias
            y_predicted = self.sigmoid(model)
            # update weights:
            # w=w-a*dw
            # b=b-a*db
            dw = 1 / (n_samples) * np.dot(X.T, (y_predicted - y))
            db = 1 / (n_samples) * np.sum(y_predicted - y)
            self.weigths -= self.learning_rate * dw
            self.bias = self.bias - self.bias * db

    def make_prediction(self, X):
        linear_model = np.dot(X, self.weigths) + self.bias
        predictions = self.sigmoid(linear_model)
        predicted_class = [1 if i > 0.5 else 0 for i in predictions]
        return predicted_class

    def sigmoid(self, c):
        return 1 / (1 + np.exp(-c))
