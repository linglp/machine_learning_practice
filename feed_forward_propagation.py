import numpy as np

def sigmoid(z):
    """sigmoid activation function on input z"""
    return 1 / (1 + np.exp(-z)) # defines the sigmoid activation function

def forward_propagation(X, Y, W1, b1, W2, b2):
    """
     Computes the forward propagation operation of a perceptron and 
     returns the output after applying the step activation function
    """
    net_h = np.dot(W1, X) + b1 # net output at the hidden layer
    out_h = sigmoid(net_h) # actual after applying sigmoid 
    net_y = np.dot(W2, out_h) + b2 # net output at the output layer
    out_y = sigmoid(net_y) # actual output at the output layer

    return out_h, out_y


## Initializing parameters
np.random.seed(42) # initializing with the same random number

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # input array
Y = np.array([[0, 1, 1, 0]]) # output label
n_h = 2 # number of neurons in the hidden layer
n_x = X.shape[0] # number of neurons in the input layer
n_y = Y.shape[0] # number of neurons in the output layer
W1 = np.random.randn(n_h, n_x) # weights from the input layer
b1 = np.zeros((n_h, 1)) # bias in the hidden layer
W2 = np.random.randn(n_y, n_h) # weights from the hidden layer
b2 = np.zeros((n_y, 1)) # bias in the output layer

# Compute forward propagation pass
A1, A2 = forward_propagation(X, Y, W1, b1, W2, b2)

pred = (A2 > 0.5) * 1
print("Predicted label:", pred) # the predicted value