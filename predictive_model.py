from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Study hours vs marks dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([40, 50, 55, 65, 70, 80, 90, 95])

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
predicted_marks = model.predict([[9]])

print("Predicted Marks for 9 Hours of Study:", predicted_marks[0])

# Graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Study Hours vs Marks Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()