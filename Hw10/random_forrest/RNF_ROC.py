import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def compute_roc_auc(train_path, dev_path, eval_path, n_trees):
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

    # Predict probabilities for ROC curve
    y_train_score = clf.predict_proba(X_train)[:, 1]
    y_dev_score = clf.predict_proba(X_dev)[:, 1]
    y_eval_score = clf.predict_proba(X_eval)[:, 1]

    # Plot ROC curve
    plot_roc_curve(y_eval, y_eval_score)

    # Calculate AUC
    auc_train = roc_auc_score(y_train, y_train_score)
    auc_dev = roc_auc_score(y_dev, y_dev_score)
    auc_eval = roc_auc_score(y_eval, y_eval_score)

    return auc_train, auc_dev, auc_eval

# Paths to datasets
train_path = './set10/train.csv'
dev_path = './set10/dev.csv'
eval_path = './set10/eval.csv'

# Number of trees
n_trees = 11

# Compute ROC curve and AUC
auc_train, auc_dev, auc_eval = compute_roc_auc(train_path, dev_path, eval_path, n_trees)

# Print AUC
print("AUC on train set:", auc_train)
print("AUC on dev set:", auc_dev)
print("AUC on eval set:", auc_eval)
