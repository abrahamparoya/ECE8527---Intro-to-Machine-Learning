import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv('./set10/train.csv')
dev_data = pd.read_csv('./set10/dev.csv')
eval_data = pd.read_csv('./set10/eval.csv')

# Separate features and target variable
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

X_dev = dev_data.drop('class', axis=1)
y_dev = dev_data['class']

X_eval = eval_data.drop('class', axis=1)
y_eval = eval_data['class']

# Initialize a list to store error rates
error_rates = []

# Try different numbers of trees
for n_trees in range(1, 101, 5):
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    clf.fit(X_train, y_train)
    eval_predictions = clf.predict(X_eval)
    eval_accuracy = accuracy_score(y_eval, eval_predictions)
    error_rate = 1 - eval_accuracy
    error_rates.append(error_rate)

# Plot error rates
plt.plot(range(1, 101,5), error_rates, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Number of Trees')
plt.grid(True)
plt.show()

# Find the optimal number of trees
optimal_n_trees = error_rates.index(min(error_rates)) + 1
print("Optimal number of trees for the eval set:", optimal_n_trees)
