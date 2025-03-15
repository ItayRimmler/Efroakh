"""

Script name: "main.py"\n
Goal of the script: The main script.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""

# Imports
import nn
from hyperparameters import *
from input.created_datasets.direction_embeddings import load_directions, create_directions
import torch
from input.embedding.embed import embed
import numpy as np

# Loading the data
# create_directions() # UNCOMMENT IF "directions_dataset_labels.npz" doesn't exist!
dataset, labels = load_directions()
test = dataset[-TEST(dataset.shape[0]):]
dataset = dataset[:TRAIN(dataset.shape[0])]
test_labels = labels[-TEST(labels.shape[0]):]
labels = labels[:TRAIN(labels.shape[0])]

# Defining sizes
input_layer_size = dataset.shape[-1]
output_layer_size = CLASSES
hidden_layer_size = HIDDEN_LAYER_SIZE
number_of_hidden_layers = HIDDEN_LAYER_AMOUNT

# Converting data to pytorch.tensor
test = torch.from_numpy(test).type(torch.float32)
dataset = torch.from_numpy(dataset).type(torch.float32)
test_labels = torch.from_numpy(test_labels).long()
labels = torch.from_numpy(labels).long()

# Loading model
model = nn.RNN_Model(input_layer_size, output_layer_size, hidden_layer_size, number_of_hidden_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# Training
model.train()

for j in range(EPOCHS):
    indices = torch.randperm(dataset.size()[0])[:SUBSET_SIZE]
    outputs = model(dataset[indices, :])
    loss = criterion(outputs, labels[indices])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{j + 1}/{EPOCHS}], Loss: {loss.item():.4f}')

# Testing
model.eval()

# Initializing variables
correct_predictions_amount = 0
total_samples = test_labels.size()[0]

with torch.no_grad():
    predicted = model(test)
    for i in range(total_samples):
        correct_predictions_amount += int(torch.argmax(predicted, dim=1).squeeze(0)[i].item() == test_labels[i].item())

# Printing accuracy
accuracy = correct_predictions_amount/total_samples
print(f'Accuracy: {accuracy * 100:.2f}%')

# Testing with new inputs:
while True:
    prompt = input()
    if prompt == 'exit':
        break
    new_prompt, _ = embed([prompt], None)
    new_prompt = torch.from_numpy(new_prompt).type(torch.float32)

    # Inserting it into the model
    with torch.no_grad():
        final_output = torch.nn.functional.softmax(model(new_prompt), dim=-1)
        predicted_class = torch.argmax(final_output, dim=-1)
    if predicted_class == 0:
        print("up", prompt)
    elif predicted_class == 1:
        print("down", prompt)
    elif predicted_class == 2:
        print("right", prompt)
    elif predicted_class == 3:
        print("left", prompt)
    else:
        print("none", prompt)
