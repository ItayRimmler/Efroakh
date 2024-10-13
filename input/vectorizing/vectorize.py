"""

Script name: "input/vectorizing/vectorize.py"\n
Goal of the script: Contains the main function that performs the whole vectorizing process.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Import functions
import input.vectorizing.pair as pair


def vectorize(dataset, vocab_len, window_size=2):
    new_dataset = pair.context_target_pairs(dataset, window_size, vocab_len)
    return new_dataset
