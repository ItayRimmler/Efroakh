"""

Script name: "main.py"\n
Goal of the script: The main script.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


import nn
from hyperparameters import *
from input.created_datasets.direction_embeddings import load_directions
from input.embedding.embed import embed
import numpy as np

dataset, labels, vocabulary = load_directions()
test = dataset[-TEST(dataset.shape[0]):]
dataset = dataset[:TRAIN(dataset.shape[0])]
test_labels = labels[-TEST(labels.shape[0]):]
labels = labels[:TRAIN(labels.shape[0])]
input_layer_size = dataset.shape[-1]
output_layer_size = len(vocabulary)
hidden_layer_sizes = HIDDEN_LAYER_SIZES
a_nodes, weights, bias = nn.create_NN(input_layer_size, hidden_layer_sizes, output_layer_size)

# Training
for j in range(EPOCHS):
    for i in range(dataset.shape[0]):
        a_nodes, z_nodes = nn.feedforward(a_nodes, weights, bias, dataset[i, 0, :]/np.max(dataset[i, 0, :]))
        weights, bias = nn.backpropagation_and_optimization(weights, bias, z_nodes, a_nodes, labels[i, 0, :], LR)
    loss = nn.cross_entropy_loss(a_nodes[-1], labels[labels.shape[0]-1,0,:])
    print(f"Training. Current epoch: {j + 1}/{EPOCHS}. Current loss: {loss}")

correct_predictions_amount = 0
total_loss = 0

# Testing
for i in range(test.shape[0]):
    nodes, _ = nn.feedforward(a_nodes, weights, bias, test[i, 0, :]/np.max(test[i, 0, :]))
    total_loss += nn.cross_entropy_loss(nodes[-1], test_labels[test_labels.shape[0]-1,0,:])
    if np.argmax(nodes[-1]) == np.argmax(test_labels[i, 0, :]):
        correct_predictions_amount += 1


accuracy = correct_predictions_amount/test.shape[0]
average_loss = total_loss/test.shape[0]

print(f"Acc: {accuracy}")
print(f"Avg Loss: {average_loss}")

# Testing with a new input
prompt = [input()]
new_prompt, _ = embed(prompt, None, vocabulary)
final_nodes, _ = nn.feedforward(a_nodes, weights, bias, new_prompt[0, 0, :]/np.max(new_prompt[0, 0, :]))
final_output = final_nodes[-1]
predicted_class = np.argmax(final_output)
print(list(vocabulary.keys())[list(vocabulary.values()).index(predicted_class)]) # Prints the most significant word