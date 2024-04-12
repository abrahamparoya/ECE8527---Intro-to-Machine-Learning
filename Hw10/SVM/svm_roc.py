import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from SVM import SVM

'''
# Load training and evaluation data
train_data = pd.read_csv('../set10/train.csv')
eval_data = pd.read_csv('../set10/eval.csv')

# Extract features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_eval = eval_data.iloc[:, 1:].values
y_eval = eval_data.iloc[:, 0].values

# Train SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Predict probabilities on evaluation set
y_prob = svm.predict_proba(X_eval)[:, 1]

'''

class ROC(object):
    def __init__(self):
        self.svm_data = SVM()

        self.fpr, self.tpr, self.thresholds, self.roc_auc = self.get_roc()

        self.plot_roc()

    def get_roc(self):
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(svm_data.y_eval, svm_data.y_prob)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, thresholds, roc_auc

    def plot_roc(self):
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('SVM_ROC.png')

    def plot_error_rate(self):
        # Plotting error rate vs number of support vectors
        plt.figure(figsize=(10, 6))
        plt.plot(svm_data.num_support_vectors, svm_data.error_rate, marker='o', linestyle='-')
        plt.xlabel('Number of Support Vectors')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Number of Support Vectors')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig('SVM_Error_Rate.png')

# Run the SVM
if __name__ == "__main__" :
    ROC()
