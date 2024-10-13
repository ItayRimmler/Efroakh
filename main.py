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
import input.vectorizing.vectorize as vectorize
import nn
from hyperparameters import *

corpus = txt_reader.read_prompts_from_file("./assets/prompt_dataset.txt")
dataset, vocab_len, translator = embed.embed(corpus)
new_dataset = vectorize.vectorize(dataset, vocab_len)
input_layer_size = new_dataset.shape[-1]
output_layer_size = vocab_len
hidden_layer_sizes = HIDDEN_LAYER_SIZES
a_nodes, weights, bias = nn.create_NN(input_layer_size, hidden_layer_sizes, output_layer_size)

# Training currently with the whole dataset
for j in range(EPOCHS):
    for i in range(new_dataset.shape[0]):
        a_nodes, z_nodes = nn.feedforward(a_nodes, weights, bias, new_dataset[i, 0, :])
        weights, bias = nn.backpropagation_and_optimization(weights, bias, z_nodes, a_nodes, new_dataset[i, 1, :], LR)
    loss = nn.cross_entropy_loss(a_nodes[-1], new_dataset[new_dataset.shape[0],1,:])
    print(f"Current epoch: {j + 1}/{EPOCHS}. Current loss: {loss}")

# Testing blindly:
prompt = input()
new_prompt, _ , _ = embed.embed(prompt)
final_prompt = vectorize.vectorize(new_prompt, vocab_len)
final_nodes, _ = nn.feedforward(a_nodes, weights, bias, final_prompt[0, 0, :])
final_output = final_nodes[-1]
import numpy as np
predicted_class = np.argmax(final_output)
print(list(translator.keys())[list(translator.values()).index(predicted_class)]) # Prints the most significant word



