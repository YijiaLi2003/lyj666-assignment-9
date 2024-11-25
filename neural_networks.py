import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Initialize weights using a normal distribution with default variance (std=1)
        self.W1 = np.random.randn(input_dim, hidden_dim)  # Weights from input to hidden layer
        self.b1 = np.zeros((1, hidden_dim))  # Biases for hidden layer
        self.W2 = np.random.randn(hidden_dim, output_dim)  # Weights from hidden to output layer
        self.b2 = np.zeros((1, output_dim))  # Biases for output layer

        # Variables to store activations and gradients for visualization
        self.hidden = None
        self.output = None
        self.gradients = None

    def forward(self, X):
        # Forward pass: compute activations
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear combination for hidden layer

        # Apply activation function
        if self.activation_fn == 'tanh':
            self.hidden = np.tanh(self.z1)
        elif self.activation_fn == 'relu':
            self.hidden = np.maximum(0, self.z1)
        elif self.activation_fn == 'sigmoid':
            self.hidden = 1 / (1 + np.exp(-self.z1))
        else:
            raise ValueError("Unsupported activation function")

        # Output layer (no activation function for binary classification with cross-entropy loss)
        self.z2 = np.dot(self.hidden, self.W2) + self.b2
        self.output = 1 / (1 + np.exp(-self.z2))  # Sigmoid activation for output layer

        return self.output

    def backward(self, X, y):
        m = y.shape[0]  # Number of samples

        # Compute gradients using chain rule
        dz2 = self.output - y  # Derivative of loss w.r.t. output layer pre-activation
        self.dW2 = np.dot(self.hidden.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Backpropagate through activation function in hidden layer
        if self.activation_fn == 'tanh':
            da1 = 1 - np.power(self.hidden, 2)  # Derivative of tanh
        elif self.activation_fn == 'relu':
            da1 = np.where(self.z1 > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            s = self.hidden
            da1 = s * (1 - s)
        else:
            raise ValueError("Unsupported activation function")

        dz1 = np.dot(dz2, self.W2.T) * da1
        self.dW1 = np.dot(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1

        # Store gradients for visualization (optional)
        self.gradients = {
            'dW1': self.dW1,
            'db1': self.db1,
            'dW2': self.dW2,
            'db2': self.db2
        }


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


# Visualization helpers
def plot_features(features, ax, title):
    for i in range(features.shape[1]):
        ax.plot(features[:, i], label=f"Neuron {i+1}")
    ax.set_title(title)
    ax.legend()


def plot_gradient_contours(X, y, ax, title):
    xx, yy = np.meshgrid(np.linspace(-3, 8, 100), np.linspace(-3, 8, 100))
    grad_magnitude = np.sqrt(xx**2 + yy**2)
    ax.contourf(xx, yy, grad_magnitude, levels=20, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolor="k")
    ax.set_title(title)


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.hidden  # (n_samples x hidden_dim=3)
    # Since hidden_features is 3D, we can plot it in 3D space
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    # Set title and labels for hidden space
    ax_hidden.set_title(f'Hidden space at step {frame * 10}')
    ax_hidden.set_xlabel('Neuron 1 activation')
    ax_hidden.set_ylabel('Neuron 2 activation')
    ax_hidden.set_zlabel('Neuron 3 activation')

    # Input space
    # Plot data points
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title(f'Input space at step {frame * 10}')
    ax_input.set_xlabel('x1')
    ax_input.set_ylabel('x2')

    # Decision boundary in input space
    # Create a meshgrid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid).reshape(xx.shape)
    ax_input.contour(xx, yy, probs, levels=[0.5], cmap="Greys_r")

    # Network structure visualization
    ax_gradient.set_title(f'Network structure at step {frame * 10}')
    # Nodes: Input layer x1, x2; Hidden layer h1, h2, h3; Output y
    # Positions for nodes
    node_pos = {
        'x1': (0, 1),
        'x2': (0, -1),
        'h1': (2, 2),
        'h2': (2, 0),
        'h3': (2, -2),
        'y': (4, 0)
    }
    # Draw nodes
    for node, (x_pos, y_pos) in node_pos.items():
        circle = Circle((x_pos, y_pos), 0.2, color='lightblue', ec='black', zorder=5)
        ax_gradient.add_patch(circle)
        ax_gradient.text(x_pos, y_pos, node, ha='center', va='center', zorder=10)
    ax_gradient.set_xlim(-1, 5)
    ax_gradient.set_ylim(-3, 3)
    ax_gradient.axis('off')
    # Draw edges with thickness representing weights
    # Input to Hidden layer edges
    for i, input_node in enumerate(['x1', 'x2']):
        for j, hidden_node in enumerate(['h1', 'h2', 'h3']):
            weight = mlp.W1[i, j]
            lw = np.clip(np.abs(weight), 0.1, 5)  # Line width proportional to weight magnitude
            color = 'red' if weight < 0 else 'green'
            x_coords = [node_pos[input_node][0], node_pos[hidden_node][0]]
            y_coords = [node_pos[input_node][1], node_pos[hidden_node][1]]
            ax_gradient.plot(x_coords, y_coords, color=color, linewidth=lw)

    # Hidden to Output layer edges
    for j, hidden_node in enumerate(['h1', 'h2', 'h3']):
        weight = mlp.W2[j, 0]
        lw = np.clip(np.abs(weight), 0.1, 5)
        color = 'red' if weight < 0 else 'green'
        x_coords = [node_pos[hidden_node][0], node_pos['y'][0]]
        y_coords = [node_pos[hidden_node][1], node_pos['y'][1]]
        ax_gradient.plot(x_coords, y_coords, color=color, linewidth=lw)


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input,
                                     ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"  
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
