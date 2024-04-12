import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss

class SVM(object):
    def __init__(self):
        # Load training and evaluation data
        self.train_data = pd.read_csv('../set10/train.csv')
        self.eval_data = pd.read_csv('../set10/eval.csv')

        # Extract features and labels
        self.X_train = self.train_data.iloc[:, 1:].values
        self.y_train = self.train_data.iloc[:, 0].values
        self.X_eval = self.eval_data.iloc[:, 1:].values
        self.y_eval = self.eval_data.iloc[:, 0].values

        # Initialize variables to store error rate and number of support vectors
        self.error_rate, self.num_support_vectors, self.y_pred, self.y_prob = self.best_support_vectors()

        # plot the error rate
        #self.plot_error_rate()

    def best_support_vectors(self):
        # Initialize variables to store error rate and number of support vectors
        error_rate = []
        num_support_vectors = []

        # Loop through different C values to find the best number of support vectors
        C_values = np.logspace(-3, 2, 20)  # try different regularization parameters
        #C_values = [1]
        for C in C_values:
            # Train SVM classifier
            svm = SVC(kernel='rbf', C=C, probability=True)
            svm.fit(self.X_train, self.y_train)
            
            # Evaluate on evaluation set
            y_pred = svm.predict(self.X_eval)
            # Predict probabilities on evaluation set
            y_prob = svm.predict_proba(self.X_eval)[:, 1]
            error = zero_one_loss(self.y_eval, y_pred)
            print("For C Value {0} error rate is {1}".format(C, error))
            error_rate.append(error)
            num_support_vectors.append(np.sum(svm.n_support_))

        return error_rate, num_support_vectors, y_pred, y_prob

    def plot_error_rate(self):
        # Plotting error rate vs number of support vectors
        plt.figure(figsize=(10, 6))
        plt.plot(self.num_support_vectors, self.error_rate, marker='o', linestyle='-')
        plt.xlabel('Number of Support Vectors')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Number of Support Vectors')
        plt.xscale('log')
        plt.grid(True)
        plt.show()

# Run the SVM
if __name__ == "__main__" :
    SVM()
