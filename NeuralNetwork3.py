import numpy as np
import sys
import math
from datetime import datetime

# read input
def read_input():
    if len(sys.argv) == 4:
        train_image = sys.argv[1]
        train_label = sys.argv[2]
        test_image = sys.argv[3]
    else:
        train_image = "train_image.csv"
        train_label = "train_label.csv"
        test_image = "test_image.csv"

    return train_image, train_label, test_image

# convert input into desired format for input layer
def format_input(train_image, train_label, test_image):
    train_image_arr = np.genfromtxt(train_image, delimiter=',')
    train_image_arr = train_image_arr/255
    train_image_arr_rows = train_image_arr.shape[0]

    train_label_arr = np.genfromtxt(train_label, delimiter=',')
    train_label_arr = train_label_arr.reshape(1,train_image_arr_rows)
    train_label_arr_new = np.eye(10)[train_label_arr.astype('int32')]
    train_label_arr_new = train_label_arr_new.T.reshape(10,train_image_arr_rows)

    test_image_arr = np.genfromtxt(test_image, delimiter=',')

    train_image_arr = train_image_arr.T
    test_image_arr = test_image_arr.T

    return train_image_arr, train_label_arr_new, test_image_arr

def write_output(prediction):
    np.savetxt("test_predictions.csv", prediction, delimiter=",", fmt="%d")

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))              
    return s

def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x), axis=0)
    return s

def compute_loss(X, Y):
    L_sum = np.sum(np.multiply(X, np.log(Y)))
    m = X.shape[1]
    L = -(1/m) * L_sum

    return L

def forward_propagation(X, params):
    cache = {}

    # LINEAR-->SIGMOID-->LINEAR-->SIGMOID-->LINEAR-->SOFTMAX
    cache['Z1'] = np.matmul(params['W1'], X) + params['b1']
    cache['A1'] = sigmoid(cache['Z1'])
    cache['Z2'] = np.matmul(params['W2'], cache['A1']) + params['b2']
    cache['A2'] = sigmoid(cache['Z2'])
    cache['Z3'] = np.matmul(params['W3'], cache['A2']) + params['b3']
    cache['A3'] = softmax(cache['Z3'])

    return cache

def backward_propagation(X, Y, params, cache):

    dZ3 = cache['A3'] - Y
    m3 = cache['A2'].shape[1]
    dW3 = (1. / m3) * np.matmul(dZ3, cache['A2'].T)
    db3 = (1. / m3) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.matmul(params["W3"].T, dZ3)
    dZ2 = dA2 * sigmoid(cache['Z2']) * (1 - sigmoid(cache['Z2']))
    m2 = cache['A1'].shape[1]
    dW2 = (1. / m2) * np.matmul(dZ2, cache['A1'].T)
    db2 = (1. / m2) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params['W2'].T, dZ2)
    dZ1 = dA1 * sigmoid(cache['Z1']) * (1 - sigmoid(cache['Z1']))
    m1 = X.shape[1]
    dW1 = (1. / m1) * np.matmul(dZ1, X.T)
    db1 = (1. / m1) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2, 'dW3':dW3, 'db3':db3}

    return gradients

def get_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    mini_batches = list()
    no_batches = math.floor(m / batch_size)
    for i in range(0, no_batches):
        batchX = X[:, i * batch_size : (i+1) * batch_size]
        batchY = Y[:, i * batch_size : (i+1) * batch_size]
        mini_batch = (batchX, batchY)
        mini_batches.append(mini_batch)

    if m % batch_size != 0:
        batchX = X[:, batch_size * math.floor(m / batch_size) : m]
        batchY = Y[:, batch_size * math.floor(m / batch_size) : m]
        mini_batch = (batchX, batchY)
        mini_batches.append(mini_batch)

    return mini_batches

def train_model(trainX, trainY, testX):

    np.random.seed(138)

    input_nodes = trainX.shape[0]
    h1_nodes = 512
    h2_nodes = 64
    alpha = 4              
    batch_size = 64
    epochs = 20
    digits = 10
    beta = 0.9  

    #Xavier initialization for parameters
    params = {
        'W1' : np.random.randn(h1_nodes, input_nodes) * np.sqrt(1.0 / input_nodes),
        'b1' : np.zeros((h1_nodes, 1)) * np.sqrt(1.0 / input_nodes),
        'W2' : np.random.randn(h2_nodes, h1_nodes) * np.sqrt(1.0 / h1_nodes),
        'b2' : np.zeros((h2_nodes, 1)) * np.sqrt(1.0 / h1_nodes),
        'W3' : np.random.randn(digits, h2_nodes) * np.sqrt(1.0 / h2_nodes),
        'b3' : np.zeros((digits, 1)) * np.sqrt(1.0 / h2_nodes)
    }

    #parameter derivates initialization
    m_dW1 = np.zeros(params['W1'].shape)
    m_db1 = np.zeros(params['b1'].shape)
    m_dW2 = np.zeros(params['W2'].shape)
    m_db2 = np.zeros(params['b2'].shape)
    m_dW3 = np.zeros(params['W3'].shape)
    m_db3 = np.zeros(params['b3'].shape)

    # training the model
    for x in range(epochs):
        perm = np.random.permutation(trainX.shape[1])    
        trainX_random = trainX[:, perm]    
        trainY_random = trainY[:, perm]

        mini_batches = get_mini_batches(trainX_random, trainY_random, batch_size)

        for mini_batch in mini_batches:
            X, Y = mini_batch
            cache = forward_propagation(X, params)
            grad = backward_propagation(X, Y, params, cache)

            m_dW1 = (beta * m_dW1 + (1. - beta) * grad['dW1'])
            m_db1 = (beta * m_db1 + (1. - beta) * grad['db1'])
            m_dW2 = (beta * m_dW2 + (1. - beta) * grad['dW2'])
            m_db2 = (beta * m_db2 + (1. - beta) * grad['db2'])
            m_dW3 = (beta * m_dW3 + (1. - beta) * grad['dW3'])
            m_db3 = (beta * m_db3 + (1. - beta) * grad['db3'])

            #update the paramaters using SGD with momentum
            params['W1'] = params['W1'] - alpha * m_dW1
            params['b1'] = params['b1'] - alpha * m_db1
            params['W2'] = params['W2'] - alpha * m_dW2
            params['b2'] = params['b2'] - alpha * m_db2
            params['W3'] = params['W3'] - alpha * m_dW3
            params['b3'] = params['b3'] - alpha * m_db3

        # cache = forward_propagation(trainX, params)
        # output = cache['A3']
        # acc = []
        # p = np.argmax(output, axis=0)
        # acc.append(p == np.argmax(trainY, axis=0))
        # accuracy = np.mean(acc)
        # training_cost = compute_loss(trainY, cache['A3'])
        # print("Epoch {} : Training cost = {}".format(x+1, training_cost))
        # print("Epoch {} : Accuracy = {}".format(x+1, accuracy * 100))

    
    # prediction on test data
    cache = forward_propagation(testX, params)
    prediction = np.argmax(cache['A3'], axis=0)

    # correct = 0
    # x_shape = testX.shape[1]
    # y_test = np.genfromtxt('test_label.csv', delimiter=',')
    # y_test = y_test.reshape(1, x_shape)
    # y_new_test = np.eye(10)[y_test.astype('int32')]
    # y_new_test = y_new_test.T.reshape(10, x_shape)
    # labels = np.argmax(y_new_test, axis=0)
    # for i in range(len(prediction)):
    #     if prediction[i] == labels[i]:
    #        correct = correct + 1
    # print("Number of correctly classified samples (out of 10000) : ", correct)

    return prediction

start = datetime.now()
if __name__ == '__main__':
    train_image, train_label, test_image = read_input()
    trainX, trainY, testX = format_input(train_image, train_label, test_image)
    prediction = train_model(trainX, trainY, testX)
    write_output(prediction)
end = datetime.now()
print("time taken = ", end - start)