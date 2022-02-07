import sklearn.metrics as metrics
# def calculate_error(y, y_predicted):
#    """Calculate the cross entropy losss"""
#    loss = np.sum(- y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted))
#    return loss

def forward_propagation(X, W, b):
    """
     Computes the forward propagation operation of a perceptron and 
     returns the output after applying the step activation function
    """
    weighted_sum = np.dot(X, W) + b # calculate the weighted sum of X and W
    return weighted_sum

def gradient(X, Y, Y_predicted):
    """"Gradient of weights and bias"""
    Error = Y_predicted - Y # Calculate error
    dW = np.dot(X.T, Error) # Compute derivative of error w.r.t weight, i.e., (target - output) * x
    db = np.sum(Error) # Compute derivative of error w.r.t bias
    return dW, db # return derivative of weight and bias

def update_parameters(W, b, dW, db, learning_rate):
    """Updating the weights and bias value"""
    W = W - learning_rate * dW # update weight
    b = b - learning_rate * db # update bias
    return W, b # return weight and bias

def train(X, y, w, b, epochs, l_r):
   # your code goes here!
   #loss = np.zeros((epochs, 1))

   for i in range(epochs):
      Y_predicted = np.dot(X, w) + b
      #MSE = np.square(np.subtract(y,Y_predicted)).mean()
      MSE = metrics.mean_squared_error(y, Y_predicted)/2
      dW, db = gradient(X, y, Y_predicted)
      w, b = update_parameters(w, b, dW, db, l_r) # update parameters

   return w, b, MSE