"""

Script name: "nn.py"\n
Goal of the script: Contains functions used to build and train a NN.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Imports
import numpy as np
from hyperparameters import *

# Functions
def create_random_matrix(my_size, my_range):
    """
    Creates a matrix* of random numbers**\n
    :param my_size: *this is the size
    :param my_range: **this is the range in which the numbers will be
    :return: The said matrix
    """
    return np.random.rand(my_size[0], my_size[1]) * (max(my_range) - min(my_range)) + min(my_range)

def softmax(x):
    """
    Softmax activation function for an element** in a vector*\n
    :param x: *this is the vector
    :param i: **this is the index of the element
    :return: Node's value after applying the activation function on it
    """
    numerator = np.exp(x - np.max(x))
    # denominator = 0
    # for element in x:
    #     denominator += np.sum(np.exp(element))
    return numerator/np.sum(numerator)

def ReLU(x):
    """
    LEAKY ReLU.\n
    """
    if x > 0:
        return x
    return LEAK * x

def feedforward(nodes, weights, bias, input):
    """
    Feeds forward the NN:\n
    1) Initializes the nodes* to be the input&\n
    2) Initializes activated_nodes to be input&\n
    3) Iterating on each layer other than the input layer\n
    4) Performing weights** transposed dot the previous layer's nodes, and saving it in the current layer's nodes\n
    5) Adding the bias*** to the nodes\n
    6) Inside activated_nodes, we save a new uninitialized layer (a vector of zeros)\n
    7) Each node in nodes* in the relevant layer we activate, then we save the value in the new layer in activated_nodes. The activated node value isn't saved in nodes*\n
    NOTE: Currently the activation function is my own implemented ReLU for the hidden layers, and my own implemented softmax for the last layer.\n
    :param nodes: *a list of np arrays in different sizes that represent the values of the nodes in each layer: the input, hidden, and output layers
    :param weights: **a list of weight matrices of the different layers
    :param bias: ***a list of bias vectors of the different layers
    :param input: &a vector of values we insert into the input layer
    :return: activated_nodes, a list in the same length of nodes*, and the same np arrays it contains, but with different values. Also we return nodes that went through everything but the activation function. Later, we shall use both sets of nodes in the backpropagation
    """
    nodes[0] = input
    activated_nodes = [nodes[0]]
    for i in range(1, len(weights) + 1):
        nodes[i] = np.dot(weights[i - 1], nodes[i - 1])
        nodes[i] += bias[i] # Were activated in the last layer, but we're moving a layer so now they're just regular nodes...
        activated_nodes.append(np.zeros(nodes[i].shape))
        for j in range(activated_nodes[-1].shape[0]):
            if i == len(weights):
                activated_nodes[i] = softmax(nodes[i])
                break
            else:
                activated_nodes[i][j] = ReLU(nodes[i][j])
    return activated_nodes, nodes

def create_NN(input, layers, output, bias=None, w_range=None):
    """
    Creates a NN with random weights:\n
    1) Creates a list of matrices called "weights" in size "input"*, "output"*** and all the sizes in "layers"**\n
    2) Fills each weight matrix in numbers in the range of 0 to 2/sqrt(number_of_columns_in_the_matrix), or in the range w_range&&\n
    3) Sets the nodes in the input layer to initial_input, if it doesn't exist then it will be random numbers between 0 and 1\n
    4) Does the same with the biases vector. If it's given, then we're using the variable bias&, else we use a variable called bias_list, unused in case of existing bias&\n
    :param input: *integer, indicates how big is the input vector
    :param layers: **a list of integers, the size of the list is the number of hidden layers and each element is the size of each element
    :param output: ***integer, indicates how big is the output vector
    :param bias: &a pre-existing vector of values of the input layer
    :param w_range: &&a pre-existing range type variable, it will be the range of the values of the initial weights
    :return: The list of the unactivated (often notated as 'z') nodes, the list of weight matrices and the list of biases vectors
    """
    weights = []
    for i in range(len(layers)):
        if i == 0:
            if w_range:
                weights.append(create_random_matrix((layers[i], input), w_range))
            else:
                weights.append(create_random_matrix((layers[i], input), [0, np.sqrt(2/input)]))
        else:
            if w_range:
                weights.append(create_random_matrix((layers[i], layers[i-1]), w_range))
            else:
                weights.append(create_random_matrix((layers[i], layers[i - 1]), [0, np.sqrt(2/input)]))
    if w_range:
       weights.append(create_random_matrix((output, layers[-1]), w_range))
    else:
            weights.append(create_random_matrix((output, layers[-1]), [0, np.sqrt(2/input)]))
    if not bias:
        bias_list = []
    nodes = [np.zeros(input)]
    if not bias:
        bias_list = [np.zeros(input)]
    for l in layers:
        nodes.append(np.zeros(l))
        if not bias:
            bias_list.append(np.zeros(l))
    nodes.append(np.zeros(output))
    if not bias:
        bias_list.append(np.zeros(output))
    if not bias:
        return nodes, weights, bias_list
    return  nodes, weights, bias

def cross_entropy_loss(output, actual_output):
    """
    Calculates loss with the cross entropy formula\n
    You can use this solely for the purpose of tracking how well the NN is learning. This function serves no other purpose\n
    :param output: The predicted output
    :param actual_output: The actual output
    :return: The loss
    """
    return -np.sum(np.log(output) * actual_output)

def softmax_gradient(s):
    """
    Calculates the softmax gradient for later use in the backpropagation process.\n
    Credit to "akuiper" in the comments in this post: https://stackoverflow.com/questions/45949141/compute-a-jacobian-matrix-from-scratch-in-python\n
    :param s: The vector we calculate the gradient of
    :return: The gradient of the softmax
    """
    s = s.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def ReLU_gradient(x):
    return np.where(x > 0, 1,   LEAK)

def SGD(somethings_gradient, lr):
    """
    Performs Stochastic Gradient Descent\n
    :param somethings_gradient: A gradient
    :param lr: learning rate
    """
    return -lr * somethings_gradient

def backpropagation_and_optimization(weights, bias, nodes, activated_nodes,  actual_output, learning_rate):
    """
    Performs backpropagation and optimization:\n
    1) Calculates the first error term from the last layer, the one that has the output, then starts a loop\n
    2) Backpropagates to extract the error term of the current layer\n
    3) Calculates gradients of the weights* and the bias**\n
    4) Optimizes the weights and biases\n
    NOTE: Currently tailored for cross entropy loss and softmax activation in the last layer and ReLU in the hidden layers, also SGD optimizer\n
    :param weights: *list of matrices of weights of each layer
    :param bias: **list of vectors of the biases of each layer
    :param nodes: List of vectors of the nodes before applying activation function on them (even though they're created from an activated layer, they're themselves are not)
    :param activated_nodes: List of vectors of the nodes after applying activation function on them
    :param actual_output: The actual output, rather than the predicted one (activated_nodes[-1])
    :param learning_rate: Learning rate of the model
    :return: The new weight matrices list and biases vectors list
    """
    current_layer_error_term = activated_nodes[-1] - actual_output # SPECIFICALLY FOR CROSS ENTROPY LOSS AND SOFTMAX!
    for i in reversed(range(1, len(weights) + 1)):
        current_layer_error_term = np.dot(weights[i - 1].T, current_layer_error_term) * ReLU_gradient(nodes[i - 1])
        b_grad = np.sum(current_layer_error_term)
        b_grad = clip_gradient(b_grad)
        w_grad = np.dot(activated_nodes[i - 1], current_layer_error_term)
        w_grad = clip_gradient(w_grad)

        # Optimization
        bias[i - 1] += SGD(b_grad, learning_rate)
        weights[i - 1] += SGD(w_grad, learning_rate)

    return weights, bias


def clip_gradient(gradients):
    """
    Clips the gradients by the L2 norm.
    :param gradients: numpy array, the gradients to be clipped
    :return: numpy array, the clipped gradients
    """
    # Compute L2 norm of the gradients
    grad_norm = np.linalg.norm(gradients)

    # If the gradient norm exceeds the threshold, scale the gradients
    if grad_norm > CLIP:
        gradients = gradients * (CLIP / grad_norm)

    return gradients