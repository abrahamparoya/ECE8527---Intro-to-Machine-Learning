import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def compute_error_rates(train_path, dev_path, eval_path, n_trees):
    # Load datasets
    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)
    eval_data = pd.read_csv(eval_path)

    # Separate features and target variable
    X_train = train_data.drop('class', axis=1)
    y_train = train_data['class']

    X_dev = dev_data.drop('class', axis=1)
    y_dev = dev_data['class']

    X_eval = eval_data.drop('class', axis=1)
    y_eval = eval_data['class']

    # Initialize classifier with specified number of trees
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on train, dev, and eval sets
    train_predictions = clf.predict(X_train)
    dev_predictions = clf.predict(X_dev)
    eval_predictions = clf.predict(X_eval)

    # Compute error rates
    train_error = 1 - accuracy_score(y_train, train_predictions)
    dev_error = 1 - accuracy_score(y_dev, dev_predictions)
    eval_error = 1 - accuracy_score(y_eval, eval_predictions)

    return train_error, dev_error, eval_error

# Paths to datasets
train_path = './set10/train.csv'
dev_path = './set10/dev.csv'
eval_path = './set10/eval.csv'

# Number of trees
n_trees = 50

# Compute error rates
train_error, dev_error, eval_error = compute_error_rates(train_path, dev_path, eval_path, n_trees)

# Print error rates
print("Error rate on train set:", train_error)
print("Error rate on dev set:", dev_error)
print("Error rate on eval set:", eval_error)
