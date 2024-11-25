import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def evaluate_loss(X, y, w, b):
    predictions = X * w + b
    return np.sum((y - predictions) ** 2)

# 1. Generate synthetic data
num_points = 50
w_true = 18.6
b_true = 8.2
X = np.linspace(1, num_points, num_points)
noise = np.random.randn(num_points) * 5  # Gaussian noise
y = X * w_true + b_true + noise

# 2. Defining the solution space beforehand for initial starting points
w_range = np.linspace(w_true - 10, w_true + 10, 100)
b_range = np.linspace(b_true - 10, b_true + 10, 100)
W, B = np.meshgrid(w_range, b_range)

X_broadcasted = X[np.newaxis, np.newaxis, :]  # Shape (1,1,num_points)
Y_broadcasted = y[np.newaxis, np.newaxis, :]  
W_expanded = W[:, :, np.newaxis]  # Shape (100,100,1)
B_expanded = B[:, :, np.newaxis]  

predictions = W_expanded * X_broadcasted + B_expanded  # Shape (100,100,num_points)
Loss = np.sum((Y_broadcasted - predictions) ** 2, axis=2)  # Shape (100,100)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, Loss, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
