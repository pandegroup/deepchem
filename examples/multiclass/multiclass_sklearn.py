import deepchem as dc
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

N = 100
n_feat = 5
n_classes = 3
X = np.random.rand(N, n_feat)
y = np.random.randint(n_classes, size=(N,))
dataset = dc.data.NumpyDataset(X, y)

sklearn_model = RandomForestClassifier(class_weight="balanced", n_estimators=50)
model = dc.models.SklearnModel(sklearn_model)

# Fit trained model
print("About to fit model")
model.fit(dataset)
model.save()

print("About to evaluate model")
train_scores = model.evaluate(dataset, sklearn.metrics.roc_auc_score, [],n_classes=n_classes)

print("Train scores")
print(train_scores)
