"""

Script name: "main.py"\n
Goal of the script: The main script.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


import input.dataset_readers.txt_reader as txt_reader
import input.embedding.embed as embed
import nn
from hyperparameters import *

corpus = txt_reader.read_prompts_from_file("./assets/prompt_dataset.txt")[BEGINNING:BEGINNING + SAMPLE_SIZE]
labels = txt_reader.read_prompts_from_file("./assets/prompt_labels.txt")[BEGINNING:BEGINNING + SAMPLE_SIZE]
vocabulary = {"up":0, "down":1 ,"right":2, "left":3}
dataset, labels = embed.embed(corpus, labels, vocabulary)
input_layer_size = dataset.shape[-1]
output_layer_size = len(vocabulary)
hidden_layer_sizes = HIDDEN_LAYER_SIZES
a_nodes, weights, bias = nn.create_NN(input_layer_size, hidden_layer_sizes, output_layer_size)

# Training currently with the whole dataset
for j in range(EPOCHS):
    for i in range(dataset.shape[0]):
        a_nodes, z_nodes = nn.feedforward(a_nodes, weights, bias, dataset[i, 0, :])
        weights, bias = nn.backpropagation_and_optimization(weights, bias, z_nodes, a_nodes, labels[i, 0, :], LR)
    loss = nn.cross_entropy_loss(a_nodes[-1], labels[labels.shape[0]-1,0,:])
    print(f"Current epoch: {j + 1}/{EPOCHS}. Current loss: {loss}")

# Testing blindly:
prompt = [input()]
new_prompt, _ = embed.embed(prompt, None, vocabulary)
final_nodes, _ = nn.feedforward(a_nodes, weights, bias, new_prompt[0, 0, :])
final_output = final_nodes[-1]
import numpy as np
predicted_class = np.argmax(final_output)
print(list(vocabulary.keys())[list(vocabulary.values()).index(predicted_class)]) # Prints the most significant word



