"""

Script name: "./input/created_datasets/direction_embeddings.py"\n
Goal of the script: Functions that deal with directions' dataset.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""

# Libraries and imports
import input.dataset_readers.txt_reader as txt_reader
import input.embedding.embed as embed
import numpy as np

def create_directions():
    """
    Creates dataset and labels (without dividing into train and test) for training of the NN to detect directions.\n
    1) Loads the dataset and labels from the right directory (main.py should be in the root).\n
    2) Embeds each sentence. Embeds the labels, in this case, in a very simple manner, inside the "prepare_result_matrix" function inside the "embed" function.\n
    3) Saves a .npz file that can be loaded with numpy.\n
    NOTE: Works with Bert, usually takes like 5 minutes to run. Still, you want to run this only once.
    """
    corpus = txt_reader.read_prompts_from_file("assets/prompts.txt")
    labels = txt_reader.read_prompts_from_file("assets/labels.txt")
    dataset, labels = embed.embed(corpus, labels)
    np.savez('input/created_datasets/directions_dataset_labels.npz', dataset=dataset, labels=labels)

def load_directions():
    """
    Loads the pre-made .npz file that contains the matrix of embedded data, and the vocabulary.\n
    """
    data = np.load('./input/created_datasets/directions_dataset_labels.npz')
    dataset = data['dataset']
    labels = data['labels']
    return dataset, labels