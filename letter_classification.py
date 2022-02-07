import numpy as np 
import matplotlib.pyplot as plt 

# Creating data set   
# A 
a = [0, 0, 1, 1, 0, 0, 
   0, 1, 0, 0, 1, 0, 
   1, 1, 1, 1, 1, 1, 
   1, 0, 0, 0, 0, 1, 
   1, 0, 0, 0, 0, 1] 
   
# B 
b =[0, 1, 1, 1, 1, 0, 
   0, 1, 0, 0, 1, 0, 
   0, 1, 1, 1, 1, 0, 
   0, 1, 0, 0, 1, 0, 
   0, 1, 1, 1, 1, 0] 
# C 
c =[0, 1, 1, 1, 1, 0, 
   0, 1, 0, 0, 0, 0, 
   0, 1, 0, 0, 0, 0, 
   0, 1, 0, 0, 0, 0, 
   0, 1, 1, 1, 1, 0] 
  
# Creating labels 
y =[[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] 


# visualizing the data, plotting A 
# plt.imshow(np.array(a).reshape(5, 6)) 
# plt.show()

# # visualizing the data, plotting B
# plt.imshow(np.array(b).reshape(5, 6)) 
# plt.show()

# # visualizing the data, plotting C
# plt.imshow(np.array(c).reshape(5, 6)) 
# plt.show()



# # converting data and labels into numpy array 
# x = np.array([a, b, c]) 
# # Labels are also converted into NumPy array 
# y = np.array(y) 

# np.random.seed(42) # seed function to generate the same random value
# n_x = 30
# n_h1 = 5
# n_h2 = 4
# n_y = 3
# w1 = np.random.randn(n_x, n_h1)
# w2 = np.random.randn(n_h1, n_h2)
# w3 = np.random.randn(n_h2, n_y)
# b1 = np.zeros((1, n_h1))
# b2 = np.zeros((1, n_h2))
# b3 = np.zeros((1, n_y))
# learning_rate = 0.01
# epochs = 10000


def sigmoid(z):
    """Compute sigmoid values for each sets of scores in z"""
    return 1 / (1 + np.exp(-z))

def softmax(x):
    """Compute softmax values for each sets of scores in x"""
    return np.exp(x) / np.sum(np.exp(x), axis=1) 


def forward_propagation(x, w1, w2, w3, b1, b2, b3):
    """ 
    Computes the forward propagation operation for the 3-layered 
    neural network and returns the output at the 2 hidden layers 
    and the output layer
    """
    # net output at the first hidden layer
    net_h1 = np.dot(x, w1) + b1

    # apply sigmoid to get the atual output
    out_h1 = sigmoid(net_h1)

    # net output at the second hidden layer
    net_h2 = np.dot(out_h1, w2) + b2

    # apply sigmoid again 
    out_h2 = sigmoid(net_h2)

    # net output at the third hidden layer
    net_h3 = np.dot(out_h2, w3) + b3


    # apply the final activation function 
    out_y = softmax(net_h3)

    return out_h1, out_h2, out_y





#out_h1, out_h2, out_h3, out_y = forward_propagation(x, w1, w2, w3, b1, b2, b3)

# print("First Hidden layer output:\n", out_h1)
# print("Second Hidden layer output:\n", out_h2)
# print("Output layer:\n", out_y)


def backpropagation(y, out_y, out_h2, out_h1, w3, w2, x):
    """ 
    Computes the backpropagation operation for the 
    3-layered neural network and returns the gradients
    of weights and biases
    """
    # your code goes here!
    l3_error = out_y - y # actual - target
    dW3 = np.dot(l3_error, out_h2) # derivative of layer three weight
    db3 = np.sum(l3_error, axis = 0, keepdims=True)
    dh2 = np.dot(l3_error, w3.T) # compute dot product of weights in layer 3 with error at layer 3 
    l2_error = np.multiply(dh2, out_h2 * (1 - out_h2)) # compute layer 2 error
    dW2 = np.dot(l2_error.T, out_h1) # derivative of layer two weight
    db2 = np.sum(l2_error, axis = 0, keepdims=True)
    dh1 = np.dot(l2_error, w2.T)
    l1_error = np.multiply(dh1, out_h1 * (1 - out_h1))
    dW1 = np.dot(l1_error.T, x) 
    db1 = np.sum(l1_error, axis=0, keepdims=True)




    return dW3, db3, dW2, db2, dW1, db1 


#dW3, db3, dW2, db2, dW1, db1 = backpropagation(y, out_y, out_h2, out_h1, w3, w2, x)

# print(dW3)

# print(dW1.shape)
# print(dW2.shape)
# print(dW3.shape)
# print(db1.shape)
# print(db2.shape)



def update_parameters(w1, dW1, b1, db1, w2, dW2, b2, db2, w3, dW3, b3, db3, learning_rate):
    """Update parameters after the gradient descent operation"""
    w1 = w1 - learning_rate * dW1.T
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dW2.T
    b2 = b2 - learning_rate * db2
    w3 = w3 - learning_rate * dW3.T
    b3 = b3 - learning_rate * db3

    return w1, b1, w2, b2, w3, b3


def calculate_error(y, y_predicted):
   """Calculate the cross entropy losss"""
   loss = np.sum(- y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted))
   return loss



def train(x, y, w1, w2, w3, b1, b2, b3, epochs, learning_rate):
   """Train the 3 layered neural network"""
   losses = np.zeros((epochs, 1))
   # your code goes here!
   for i in range(1000):
       out_h1, out_h2, out_y = forward_propagation(x, w1, w2, w3, b1, b2, b3)
       losses[i, 0] = calculate_error(y, out_y)
       dW3, db3, dW2, db2, dW1, db1  =  backpropagation(y, out_y, out_h2, out_h1, w3, w2, x)
       w1, b1, w2, b2, w3, b3 = update_parameters(w1, dW1, b1, db1, w2, dW2, b2, db2, w3, dW3, b3, db3, learning_rate)
       
   return w1, b1, w2, b2, w3, b3, losses

# Evaluating the performance 
# epochs = 1000
# learning_rate = 0.5

# converting data and labels into numpy array 
x = np.array([a, b, c]) 
# Labels are also converted into NumPy array 
y = np.array(y) 

np.random.seed(42) # seed function to generate the same random value
n_x = 30 # number of nodes in the input layer
n_h1 = 5 # number of nodes in the first hidden layer
n_h2 = 4 # number of nodes in the second hidden layer
n_y = 3 # number of nodes in the output layer
w1 = np.random.randn(n_x, n_h1) # weights of the first hidden layer
w2 = np.random.randn(n_h1, n_h2) # weights of the second hidden layer
w3 = np.random.randn(n_h2, n_y) # weights of the output layer
b1 = np.zeros((1, n_h1)) # bias of the first hidden layer
b2 = np.zeros((1, n_h2)) # bias of the second hidden layer
b3 = np.zeros((1, n_y)) # bias of the output layer
epochs = 1000
learning_rate = 0.5

# Train the neural network
w1, b1, w2, b2, w3, b3, losses = train(x, y, w1, w2, w3, b1, b2, b3, epochs, learning_rate)

#print(w1, w1.shape)
# print(b1)


plt.figure() 
plt.plot(losses) 
plt.xlabel("EPOCHS") 
plt.ylabel("Loss value") 
plt.show() 
