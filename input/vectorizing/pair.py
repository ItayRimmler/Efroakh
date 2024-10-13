"""

Script name: "input/embedding/embed.py"\n
Goal of the script: Contains functions that create context-target-pairs.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Import functions
import input.vectorizing.input_vector as input_vector


# Libraries
import numpy as np


def context_target_pairs(dataset, ws, vocab_len):
    """
    Creates context-target-pairs:\n
    1) Creates new_dataset, which is a matrix of size len(dataset*)x2xvocab_len***\n
    2) Iterates over each word in the dataset, marking it as the target\n
    3) We define the context according to the ws**, the target we iterate over, and the dataset*\n
    4) In the [target's_index, 0, :] indices, there will be a one-hot-encode each context word, that has been averaged summed\n
    5) In the [target's_index, 1, :] indices, there will be a one-hot-encode of the target word\n
    :param dataset: *the original dataset, a corpus of words that's made out of integers
    :param ws: **window size of the CBOW algorithm. NOTE: The amount of context words from each side of the target
    :param vocab_len: ***the length of our vocabulary
    :return: new_dataset
    """
    new_dataset = np.zeros((len(dataset),2,vocab_len))
    for w in range(len(dataset)):
        target = input_vector.one_hot_encode(dataset[w], vocab_len)
        context = []
        left_side = max(0, w - ws)
        right_side = min(w + ws, len(dataset))
        for i in range(left_side, right_side):
            context.append(input_vector.one_hot_encode(dataset[i], vocab_len))
        the_input = np.array(input_vector.averaged_sum(context))
        the_output = np.array(target)
        new_dataset[w, 0, :] = the_input.copy()
        new_dataset[w, 1, :] = the_output.copy()
    return new_dataset