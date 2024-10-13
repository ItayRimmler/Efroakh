"""

Script name: "input/embedding/embed.py"\n
Goal of the script: Contains the main function that performs the whole embedding process.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Import functions
import input.embedding.index as index
import input.embedding.preprocess as pp


def embed(corpus):
    """
    The whole embedding process\n
    :param corpus: a bunch of strings, sentences or words that we want to analyse. NOTE: Can currently only be an iterable
    :return: The dataset, embedded
    """
    dataset = index.tokenize(corpus)
    dataset = pp.lowercase(dataset)
    dataset = pp.rid_of_i_we(dataset)
    dataset = pp.rid_of_punctuation(dataset)
    dataset = pp.rid_of_ing(dataset)
    dataset = pp.rid_of_preposition(dataset)
    vocabulary = index.map(dataset)
    dataset = index.index(dataset, vocabulary)
    return dataset, len(vocabulary), vocabulary