from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Predict for a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower
prediction = knn.predict(sample)
predicted_class = iris.target_names[prediction[0]]

print(f"Predicted Iris flower type: {predicted_class}")
