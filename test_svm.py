import svm
import numpy as np

X_train = np.array([[-1.0, -1.0], [-3.0, -3.0], [1.0, 1.0], [2.0, 2.0]])
y_train = np.array([[-1.0], [-1.0], [1.0], [1.0]])

X_test = np.array([[-2.0, -2.0], [3.0, 3.0]])
Y_test = np.array([[-1.0], [1.0]])

clsfyr = svm.train(X_train, y_train, svm.linear)
print(clsfyr)

results = svm.classify(clsfyr, X_test)
correct = 0
for r in range(len(results)):
    if results[r] == Y_test[r]:
        correct = correct + 1
print("Accuracy: ", correct / len(results))
