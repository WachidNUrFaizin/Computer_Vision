import numpy as np
import math as m
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

num_data = 1000
train_x = train_x[:num_data]
train_y = train_y[:num_data]

print(train_x.shape, train_y.shape)

# Preprocess data
np.set_printoptions(linewidth=200)
X = np.array([[[1 if dd > 0 else 0 for dd in row] for row in img] for img in train_x])

y = to_categorical(train_y)
x = np.array([img.flatten() for img in X])

# Multilayer Perceptron
input = x
target = y

num_input = input.shape[1]
num_output = target.shape[1]
num_hidden = 100

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sig = np.vectorize(sigmoid)

# Initialize weights and biases
b_hidden = np.random.uniform(low=-1, high=1, size=(num_hidden,))
w_hidden = np.random.uniform(low=-1, high=1, size=(num_hidden, num_input))

b_output = np.random.uniform(low=-1, high=1, size=(num_output,))
w_output = np.random.uniform(low=-1, high=1, size=(num_output, num_hidden))

# Learning rate
loss_values = []
acc_values = []

lr = 0.5
epochs = 1000

for epoch in range(epochs):
    MSE = 0
    new_target = np.zeros(target.shape)

    for idx, inp in enumerate(input):
        # Forward pass
        hidden = sig(np.dot(w_hidden, inp) + b_hidden)
        output = sig(np.dot(w_output, hidden) + b_output)

        # Backward pass
        error = target[idx] - output
        MSE += np.sum(error ** 2)
        delta_output = error * output * (1 - output)
        delta_hidden = hidden * (1 - hidden) * np.dot(w_output.T, delta_output)

        # Update weights and biases
        w_output += lr * np.outer(delta_output, hidden)
        b_output += lr * delta_output
        w_hidden += lr * np.outer(delta_hidden, inp)
        b_hidden += lr * delta_hidden

        new_target[idx] = output

    loss_values.append(MSE / num_data)
    acc = np.sum(np.argmax(target, axis=1) == np.argmax(new_target, axis=1)) / num_data
    acc_values.append(acc)
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {MSE / num_data} - Acc: {acc}")

plt.plot(loss_values)
plt.title("Loss")
plt.show()

plt.plot(acc_values)
plt.title("Accuracy")
plt.show()

