import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss


C_values = np.logspace(-3, 2, 20)
for C in C_values:
    print(C)