"""

Script name: "input/embedding/embed.py"\n
Goal of the script: Contains the main function that performs the whole embedding process.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""
from sipbuild.generator.parser.annotations import string

# Import functions
import input.embedding.sentence_embedding as se
import input.embedding.preprocess as pp


def embed(corpus, labels):
    """
    The whole embedding process\n
    Some work here is credited to ChatGPT.\n
    NOTE: Label embedding is done inside "se.prepare_result_matrix" function
    :param corpus: a bunch of strings, sentences or words that we want to analyse. NOTE: Can currently only be an iterable
    :return: The dataset, embedded
    """

    dataset = []

    # Preprocess
    for string in corpus:
        p_string = string.lower()
        p_string = pp.rid_of_punctuation(p_string)
        p_string = pp.rid_of_i_we(p_string)
        p_string = pp.rid_of_ing(p_string)
        p_string = pp.rid_of_preposition(p_string)
        dataset.append(p_string)

    # Load the model
    tokenizer, model = se.load_model()

    # Create embeddings for sentences
    embeddings = se.create_embeddings(dataset, tokenizer, model)

    # Prepare the result matrix
    embedding_dim = embeddings.shape[1]
    result_matrix, label_array = se.prepare_result_matrix(dataset, labels, embedding_dim, embeddings)
    return result_matrix, label_array