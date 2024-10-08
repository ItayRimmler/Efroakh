"""

Script name: "nn.py"
Goal of the script: Contains functions used to build and train a NN.
Part of project: "Efroakh"
Description of project: A video game that programs itself.
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories
Deez: Nuts

"""


# Imports
import numpy as np

# Functions
def create_random_matrix(my_size, my_range):
    """
    Creates a matrix* of random numbers**
    :param my_size: *this is the size
    :param my_range: **this is the range in which the numbers will be
    :return: The said matrix
    """
    return np.random.rand(my_size) * (max(my_range) - min(my_range)) + min(my_range)

def softmax(x, i):
    """
    Softmax activation function for an element** in a vector*
    :param x: *this is the vector
    :param i: **this is the index of the element
    :return: Node's value after applying the activation function on it
    """
    numerator = np.exp(x[i])
    denominator = np.sum(np.exp(x))
    return numerator/denominator

def feedforward(activated_nodes, weights, bias):
    """
    Feeds forward the NN:
    1) Takes the nodes* from the previous layer, performs "it dot weights** transposed"
    2) Adds to it bias***
    3) In a different variable, saves the result after passing it through an activation function
    NOTE: Currently the activation function is my own implemented softmax, it should be in the same script. If it's not then contact me
    :param activated_nodes: *these are the nodes. Note: they're "activated", the activation function has applied on them in a previous feedforward iteration during learning (unless you use this function in a non-deep learning manner and you just use this function for fun... What the fuck is wrong with you?)
    :param weights: **a list of weight matrices of the different layers
    :param bias: ***a list of bias vectors of the different layers
    :return: The list that stores node vectors that went through the new activation function, and the list that stores the node vectors without the new activation function
    """
    new_activated_nodes = np.zeros(activated_nodes.shape)
    for i in range(1, len(activated_nodes)):
        activated_nodes[i] = np.dot(np.transpose(activated_nodes[i-1]), weights[i-1])
        activated_nodes[i] += bias[i-1] # Were activated in the last layer, but we're moving a layer so now they're just regular nodes...
        new_activated_nodes[i] = softmax(activated_nodes, i)
    return new_activated_nodes, activated_nodes # a and z respectively

def create_NN(input, layers, output, bias=None, w_range=None, initial_input =None):
    """
    Creates a NN with random weights, and one feedforward execution:
    1) Creates a list of matrices called "weights" in size "input"*, "output"*** and all the sizes in "layers"**
    2) Fills each weight matrix in numbers in the range of 0 to 2/sqrt(number_of_columns_in_the_matrix), or in the range w_range&&
    3) Sets the nodes in the input layer to initial_input, if it doesn't exist then it will be random numbers between 0 and 1
    4) Does the same with the biases vector. If it's given, then we're using the variable bias&, else we use a variable called bias_list, unused in case of existing bias&
    5) Feeds forward once with the self implemented feedforward function
    :param input: *integer, indicates how big is the input vector
    :param layers: **a list of integers, the size of the list is the number of hidden layers and each element is the size of each element
    :param output: ***integer, indicates how big is the output vector
    :param bias: &a pre-existing vector of values of the input layer
    :param w_range: &&a pre-existing range type variable, it will be the range of the values of the initial weights
    :param initial_input: &&&a pre-existing vector of values of the biases
    :return: The lists of the activated and not activated (often notated as 'z') nodes, and the list of weight matrices
    """
    # Input & Output: integers
    # Layers: a list of integers
    # Bias: POTENTIALLY a list of vectors, in sizes of [input, layers[0], layers[1], ... , layers[-1], output]
    # W_range: POTENTIALLY a range
    # Initial_input: POTENTIALLY a vector of numbers
    weights = []
    for i in range(layers.shape[0]):
        if i == 0:
            if w_range:
                weights.append(create_random_matrix((layers[i], input), w_range))
            else:
                weights.append(create_random_matrix((layers[i], input), range(0, np.sqrt(2/layers[i]))))
        else:
            if w_range:
                weights.append(create_random_matrix((layers[i], layers[i-1]), w_range))
            else:
                weights.append(create_random_matrix((layers[i], layers[i - 1]), range(0, np.sqrt(2/layers[i]))))
    if w_range:
       weights.append(create_random_matrix((output, layers[-1]), w_range))
    else:
        weights.append(create_random_matrix((output, layers[-1]), range(0, np.sqrt(2/output))))
    weights = np.array(weights)
    if not bias:
        bias_list = []
    if initial_input:
        nodes = initial_input
    else:
        nodes = [np.random.randn(input)]
    if not bias:
        bias_list = [np.zeros(input)]
    for l in layers:
        nodes.append(np.zeros(l))
        if not bias:
            bias_list.append(np.zeros(l))
    nodes.append(np.zeros(output))
    if not bias:
        bias_list.append(np.zeros(output))
        activated_nodes, nodes = feedforward(nodes, weights, bias_list)
    else:
        activated_nodes, nodes = feedforward(nodes, weights, bias)
    return  activated_nodes, nodes, weights

def cross_entropy_loss(output, actual_output):
    """
    Calculates loss with the cross entropy formula
    You can use this solely for the purpose of tracking how well the NN is learning. This function serves no other purpose
    :param output: The predicted output
    :param actual_output: The actual output
    :return: The loss
    """
    return np.sum(np.log(output) * actual_output)

def softmax_gradient(s):
    """
    Calculates the softmax gradient for later use in the backpropagation process.
    Credit to "akuiper" in the comments in this post: https://stackoverflow.com/questions/45949141/compute-a-jacobian-matrix-from-scratch-in-python
    :param s: The vector we calculate the gradient of
    :return: The gradient of the softmax
    """
    s = s.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def SGD(somethings_gradient, lr):
    """
    Performs Stochastic Gradient Descent
    :param somethings_gradient: A gradient
    :param lr: learning rate
    """
    return -lr * somethings_gradient

def backpropagation_and_optimization(weights, bias, nodes, activated_nodes,  actual_output, learning_rate):
    """
    Performs backpropagation and optimization:
    1) Calculates the first error term from the last layer, the one that has the output, then starts a loop
    2) Backpropagates to extract the error term of the current layer
    3) Calculates gradients of the weights* and the bias**
    4) Optimizes the weights and biases
    NOTE: Currently tailored for cross entropy loss and softmax activation in the last layer and the hidden layers, also SGD optimizer
    :param weights: *list of matrices of weights of each layer
    :param bias: **list of vectors of the biases of each layer
    :param nodes: List of vectors of the nodes before applying activation function on them (even though they're created from an activated layer, they're themselves are not)
    :param activated_nodes: List of vectors of the nodes after applying activation function on them
    :param actual_output: The actual output, rather than the predicted one (activated_nodes[-1])
    :param learning_rate: Learning rate of the model
    :return: The new weight matrices list and biases vectors list
    """
    current_layer_error_term = activated_nodes[-1] - actual_output # SPECIFICALLY FOR CROSS ENTROPY LOSS AND SOFTMAX!
    for i in reversed(range(len(weights))):
        current_layer_error_term = np.dot(weights[i].T, current_layer_error_term) * softmax_gradient(nodes[i])
        b_grad = np.sum(current_layer_error_term)
        w_grad = np.dot(activated_nodes[i - 1], current_layer_error_term)

        # Optimization
        bias[i] += SGD(b_grad, learning_rate)
        weights[i] += SGD(w_grad, learning_rate)

    return weights, bias
