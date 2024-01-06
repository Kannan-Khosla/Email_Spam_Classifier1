import numpy as np
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score

class MultinomialNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = None
        self.feature_probs = None

    def fit(self, X, y, alpha=1.0):
        classes, class_counts = np.unique(y, return_counts=True)
        class_probs = class_counts / len(y)
        feature_probs = []

        for c in classes:
            # Select the samples with class c
            X_c = X[y == c]

            # Calculate the total count of features in class c
            total_count_c = np.sum(X_c)

            # Calculate the feature probabilities for class c
            feature_probs_c = (np.sum(X_c, axis=0) + alpha) / (total_count_c + X.shape[1] * alpha)

            feature_probs.append(feature_probs_c)

        self.classes = classes
        self.class_probs = class_probs
        self.feature_probs = feature_probs

    def predict(self, X):
        predictions = []

        for x in X:
            # Calculate the log-likelihoods for each class
            log_likelihoods = [np.sum(np.log(self.feature_probs[i]) * x) + np.log(self.class_probs[i]) for i in range(len(self.classes))]

            # Predict the class with the highest log-likelihood
            prediction = self.classes[np.argmax(log_likelihoods)]
            predictions.append(prediction)

        return np.array(predictions)

    def hyperparameter_tuning(self, X_train, y_train, X_valid, y_valid):
        alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        best_accuracy = 0
        best_alpha = None
        precision_best=0
        recall_best=0
        f1_best=0
        accuracies = []
        alphas=[]
        precisions=[]
        recalls=[]
        f1s=[]

        for alpha in alpha_values:
            self.fit(X_train, y_train, alpha=alpha)
            y_pred_valid = self.predict(X_valid)
            accuracy= np.mean(y_pred_valid == y_valid)
            precision=precision_score(y_valid, y_pred_valid, average='binary')
            recall=recall_score(y_valid, y_pred_valid, average='binary')
            f1=f1_score(y_valid, y_pred_valid, average='binary')
            recalls.append(recall)
            f1s.append(f1)
            precisions.append(precision)
            accuracies.append(accuracy)
            alphas.append(alpha)
            

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha
                precision_best=precision
                recall_best=recall
                f1_best=f1

        return best_alpha, best_accuracy,accuracies,alphas,precision_best,precisions,recall_best,recalls,f1_best,f1s


