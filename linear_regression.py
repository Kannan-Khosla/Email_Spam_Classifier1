import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score

class LinearRegression:
    def __init__(self):
        pass

    def train_linear_regression(self, X_train, y_train, X_valid, y_valid, X_test, y_test, learning_rate=0.01, epochs=1000):
        # Add a bias term to the features
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_valid_b = np.c_[np.ones((X_valid.shape[0], 1)), X_valid]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

        # Initialize weights with zeros
        theta = np.zeros((X_train_b.shape[1], 1))

        m_train = len(y_train)
        m_valid = len(y_valid)

        # Lists to store training and validation errors during each epoch
        train_errors = []
        valid_errors = []

        for epoch in range(epochs):
            # Compute predictions for training and validation sets
            predictions_train = X_train_b.dot(theta)
            predictions_valid = X_valid_b.dot(theta)

            # Compute errors for training and validation sets
            errors_train = predictions_train - y_train.reshape(-1, 1)
            errors_valid = predictions_valid - y_valid.reshape(-1, 1)

            # Compute gradients for training set
            gradients = 2/m_train * X_train_b.T.dot(errors_train)

            # Update weights
            theta = theta - learning_rate * gradients

            # Calculate training and validation errors and append to lists
            train_mse = mean_squared_error(y_train, predictions_train)
            valid_mse = mean_squared_error(y_valid, predictions_valid)

            train_errors.append(train_mse)
            valid_errors.append(valid_mse)

        # Extract weight and bias
        w = theta[1:].reshape(-1)
        b = theta[0][0]

        # Evaluate on the test set
        predictions_test = X_test_b.dot(theta)
        test_mse = mean_squared_error(y_test, predictions_test)

        return w, b, train_errors, valid_errors, test_mse

    def predict(self, X, y, w, b):
        threshold=0.5
        # Add a bias term to the features
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute predictions
        y_pred = X_b.dot(np.concatenate(([b], w)))

        # Convert predictions to binary
        y_pred_binary = (y_pred >= threshold).astype(int)
        acc=accuracy_score(y,y_pred_binary)

        return y_pred_binary,acc

    def hyperparameter_tuning_Lin(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        learning_rate_options = [0.001, 0.01, 0.05, 0.1]
        epochs_values=[50,100,1000,2000]
        best_accuracy = 0
        precision_best=0
        recall_best=0
        f1_best=0
        best_lr = None
        best_epoch_value=0
        accuracies = []
        lrs=[]
        precisions=[]
        recalls=[]
        f1s=[]
        epochs_values_list=[]

        for lr in learning_rate_options:
            for epochs in epochs_values:
                w, b, _, _, _ = self.train_linear_regression(X_train, y_train, X_valid, y_valid, X_test, y_test, lr, epochs)
                y_pred_valid, accuracy_test = self.predict(X_valid, y_valid, w, b)
                precision=precision_score(y_valid, y_pred_valid, average='binary',zero_division=1)
                recall=recall_score(y_valid, y_pred_valid, average='binary',zero_division=1)
                f1=f1_score(y_valid, y_pred_valid, average='binary',zero_division=1)
                recalls.append(recall)
                f1s.append(f1)
                precisions.append(precision)
                accuracies.append(accuracy_test)
                lrs.append(lr)
                epochs_values_list.append(epochs)

                if accuracy_test > best_accuracy:
                    best_epoch_value=epochs
                    best_accuracy = accuracy_test
                    best_lr = lr
                    precision_best=precision
                    recall_best=recall
                    f1_best=f1
                

        return best_lr, best_accuracy,precision_best,recall_best,f1_best,accuracies,lrs,precisions,recalls,f1s,epochs_values_list, best_epoch_value