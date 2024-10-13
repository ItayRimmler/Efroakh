"""

Script name: "input/vectorizing/input_vector.py"\n
Goal of the script: Contains functions that prepare the input vector itself.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Libraries
import numpy as np


def one_hot_encode(index, vocab_len):
    """
    One hot encodes an index*. For example, for index 2 and vocab_len 4: 2 -> [0, 0, 1, 0]
    :param index: *an integer that we one hot encode according to the vocab_len**0
    :param vocab_len: **length of our vocabulary
    :return: The same integer, one hot encoded
    """
    result = np.zeros(vocab_len)
    result[index] += 1
    return result


def averaged_sum(vector_list):
    result = sum(vector_list)
    size = np.sum(vector_list)
    return result/size