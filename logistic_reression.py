import numpy as np
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X, weights):
        z = np.dot(X, weights)
        return self.sigmoid(z)

    def train_logistic_regression(self,X_train, y_train, X_valid, y_valid, X_test, y_test, learning_rate=0.01, epochs=1000):
    # Add a bias term to the features
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_valid_b = np.c_[np.ones((X_valid.shape[0], 1)), X_valid]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

        # Initialize weights with zeros
        weights = np.zeros((X_train_b.shape[1], 1))

        m_train = len(y_train)
        m_valid = len(y_valid)

        # Lists to store training and validation accuracy during each epoch
        train_accuracies = []
        valid_accuracies = []

        for epoch in range(100):
            # Shuffle the training data
            X_train_b, y_train = shuffle(X_train_b, y_train, random_state=42)

            for i in range(m_train):
                # Select a random sample from the training set
                X_batch = X_train_b[i:i+1]
                y_batch = y_train[i:i+1]

                # Compute prediction
                predictions = self.predict(X_batch, weights)

                # Compute gradient for the current sample
                gradient = X_batch.T.dot(predictions - y_batch)

                # Update weights using SGD
                weights = weights - learning_rate * gradient

            # Evaluate training and validation accuracies after each epoch
            train_predictions = self.predict(X_train_b, weights)
            valid_predictions = self.predict(X_valid_b, weights)

            train_accuracy = accuracy_score(y_train, np.round(train_predictions))
            valid_accuracy = accuracy_score(y_valid, np.round(valid_predictions))

            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)

        # Extract weights and bias
        w = weights[1:].reshape(-1)
        b = weights[0][0]

        # Evaluate on the test set
        test_predictions = self.predict(X_test_b, weights)
        test_accuracy = accuracy_score(y_test, np.round(test_predictions))

        return w, b, train_accuracies, valid_accuracies, test_accuracy
    
    
    def hyperparameter_tuning(self,X_train, y_train, X_valid, y_valid, X_test, y_test):
        best_accuracy = 0
        best_epoch_value=0
        best_learning_rate=0
        learning_rates = [0.001,0.01, 0.05, 0.1]
        epochs_values=[50,100,1000,2000]
        precision_best=0
        recall_best=0
        f1_best=0
        accuracies = []
        lrs=[]
        precisions=[]
        recalls=[]
        f1s=[]
        epochs_values_list=[]

        for learning_rate in learning_rates:
            for epochs in epochs_values:
                w, b, train_accuracies, valid_accuracy, accuracy= self.train_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test, learning_rate, epochs)
                y_pred_valid = self.predict(X_valid, w)
                y_pred_valid_binary = (y_pred_valid >= 0.5).astype(int)
                precision=precision_score(y_valid, y_pred_valid_binary, average='binary',zero_division=1)
                recall=recall_score(y_valid, y_pred_valid_binary, average='binary', zero_division=1)
                f1=f1_score(y_valid, y_pred_valid_binary, average='binary', zero_division=1)
                recalls.append(recall)
                f1s.append(f1)
                precisions.append(precision)
                accuracies.append(accuracy)
                lrs.append(learning_rate)
                epochs_values_list.append(epochs)
                # Update best hyperparameters if current model is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch_value=epochs
                    best_learning_rate=learning_rate
                    precision_best=precision
                    recall_best=recall
                    f1_best=f1
                    
                    

        return best_learning_rate,best_epoch_value,epochs_values_list, best_accuracy,accuracies,lrs,precision_best,precisions,recall_best,recalls,f1_best,f1s

