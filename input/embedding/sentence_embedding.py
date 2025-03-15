"""

Script name: "./input/embedding/sentence_embedding"\n
Goal of the script: Contains functions for sentence embeddings. NOTE: You could argue that it's redundant to have this script and the embed.py script, considering we only embed sentences... But if we were to embed more things it would've been more necessary.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Imports and libraries
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


def load_model():
    """
    Load the BERT tokenizer and model.\n
    Credit to ChatGPT.\n
    :return: Tokenizer and model.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Ensure model is in evaluation mode
    return tokenizer, model


def encode_sentence(sentence, tokenizer, model):
    """
    Encode a single sentence* to get its BERT embedding.\n
    Credit to ChatGPT.\n
    :param sentence: *a string that's encoded.
    :param tokenizer: The tokenizer we're using.
    :param model: The model we're using.
    :return: The embedded sentence
    """
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Get the mean of the last hidden state (sentence embedding)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Shape: (1, 768)


def create_embeddings(sentences, tokenizer, model):
    """
    Create embeddings for sentences*.\n
    Credit to ChatGPT.\n
    :param sentences: *a list of strings, each string is a sentence
    :param tokenizer: The tokenizer we're using.
    :param model: The model we're using.
    :return: np.ndarray: Array of sentence embeddings.
    """
    embeddings = []
    for sentence in sentences:
        print(f"Embedding: {sentence}")
        embedding = encode_sentence(sentence, tokenizer, model)
        embeddings.append(embedding)
    return np.vstack(embeddings)  # Shape: (num_sentences, 768)


def prepare_result_matrix(sentences, labels, embedding_dim, embeddings):
    """
    Prepare the result matrix combining sentence embeddings and labels.\n
    Credit to ChatGPT.\n

    Parameters:
        sentences (list): List of sentences.
        labels (list): List of corresponding labels.
        embedding_dim (int): Dimension of the sentence embeddings.

    Returns:
        Resulting matrices of shape (num_sentences, embedding_dim) and (num_sentences, num_classes), containing the sentences in their embedded version and the labels in an embedded version.
    """
    num_sentences = len(sentences)
    result_matrix = np.zeros((num_sentences, embedding_dim))
    labels_array = np.zeros(num_sentences)
    for k in range(num_sentences):
        result_matrix[k, :] = embeddings[k]  # Sentence embedding
        if not labels is None:
            labels_array[k] = 0 + 1 * int(labels[k] == 'down') + 2 * int(labels[k] == 'right') + 3 * int(labels[k] == 'left') # LABEL EMBEDDING IS DONE HERE!!

    return result_matrix, labels_array
