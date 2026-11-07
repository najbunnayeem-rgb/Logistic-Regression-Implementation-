import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Input features
X = np.array([
    [100], [120], [150],
    [90], [110], [130],
    [180], [200], [220]
])

# Target labels
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


# Create OVR Logistic Regression
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)

# Train model
model.fit(X, y)


# New fruit: weight = 160 grams
new_fruit = [[160]]

# Predict probabilities
probabilities = model.predict_proba(new_fruit)

print("Probabilities:", probabilities)


# Predict final class
prediction = model.predict(new_fruit)

fruit_names = ["Apple", "Banana", "Orange"]

print("Predicted fruit:", fruit_names[prediction[0]])