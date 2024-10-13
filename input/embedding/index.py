"""

Script name: "input/embedding/index.py"\n
Goal of the script: Contains functions used to tokenize a corpus of words.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


def tokenize(corpus):
    """
    Splits the corpus* into a list of words\n
    :param corpus: *a bunch of strings, sentences or words that we want to analyse. NOTE: Can currently only be an iterable
    :return: The corpus that's now a list of words
    """
    new_corpus = []
    for string in corpus:
        list_of_words = string.split()
        for word in list_of_words:
            new_corpus.append(word)
    return new_corpus

def map(processed_corpus):
    """
    Tokenizes our processed_corpus* of words, turning each unique word to a unique integer:\n
    1) Initializing a number i, a dictionary of tokenized words, and a bank of words\n
    2) Going over each word in the corpus\n
    3) For each word in the word: we store it in the bank, and assign it a number in the dictionary\n
    4) We add 1 to i\n
    5) If the bank's size didn't grow during the insertion, then the word is already there, and we won't update i\n
    :param processed_corpus: *a corpus of words, assumed to be preprocessed already
    :return: The dictionary that contains the integer assigned to each word
    """
    i = 0
    dictionary = {}
    went_over_them = set()
    for word in processed_corpus:
        temp_size = len(went_over_them)
        went_over_them.add(word)
        if len(went_over_them) > temp_size:
            dictionary[word] = i
            i += 1
    return dictionary

def index(processed_corpus, dictionary):
    """
    Turns each word in our corpus into it's corresponding integer from the dictionary.\n
    For example: ["i", "love", "you", "you", "love", "me] will turn into [0, 1, 2, 2, 1, 3]\n
    :param processed_corpus: A corpus of words, assumed to be preprocessed already
    :param dictionary: A dictionary that contains the integer assigned to each word
    :return: The corpus, indexed
    """
    for i in range(len(processed_corpus)):
        processed_corpus[i] = dictionary[processed_corpus[i]]
    return processed_corpus