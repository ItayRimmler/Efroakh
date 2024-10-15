"""

"""

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


def load_model():
    """
    Load the BERT tokenizer and model.
    Returns:
        tokenizer: BERT tokenizer.
        model: BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Ensure model is in evaluation mode
    return tokenizer, model


def encode_sentence(sentence, tokenizer, model):
    """
    Encode a single sentence to get its BERT embedding.

    Parameters:
        sentence (str): The sentence to encode.
        tokenizer: The BERT tokenizer.
        model: The BERT model.

    Returns:
        np.ndarray: The sentence embedding.
    """
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Get the mean of the last hidden state (sentence embedding)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Shape: (1, 768)


def create_embeddings(sentences, tokenizer, model):
    """
    Create embeddings for a list of sentences.

    Parameters:
        sentences (list): List of sentences to encode.
        tokenizer: The BERT tokenizer.
        model: The BERT model.

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    embeddings = []
    for sentence in sentences:
        embedding = encode_sentence(sentence, tokenizer, model)
        embeddings.append(embedding)
    return np.vstack(embeddings)  # Shape: (num_sentences, 768)


def prepare_result_matrix(sentences, labels, embedding_dim, embeddings, num_classes, label_to_index):
    """
    Prepare the result matrix combining sentence embeddings and labels.

    Parameters:
        sentences (list): List of sentences.
        labels (list): List of corresponding labels.
        embedding_dim (int): Dimension of the sentence embeddings.

    Returns:
        np.ndarray: Resulting matrix of shape (num_sentences, 2, embedding_dim).
    """
    num_sentences = len(sentences)
    result_matrix = np.zeros((num_sentences, 1, embedding_dim))
    labels_matrix = np.zeros((num_sentences, 1, num_classes))
    for k in range(num_sentences):
        result_matrix[k, 0, :] = embeddings[k]  # Sentence embedding
        # Store the label as a one-hot encoded vector (assuming a small number of classes)
        if not labels is None:
            labels_matrix[k, 0, :] = one_hot_encode(labels[k],
                                                num_classes, label_to_index)  # Replace with your function for one-hot encoding

    return result_matrix, labels_matrix


def one_hot_encode(label, num_classes, label_to_index):
    """
    One-hot encode a label into a vector.

    Parameters:
        label (str): The label to encode.
        num_classes (int): The number of classes.

    Returns:
        np.ndarray: One-hot encoded vector.
    """
    # Create a zero array for the one-hot vector
    one_hot = np.zeros(num_classes)
    index = label_to_index[label]  # Map your label to an index
    one_hot[index] = 1
    return one_hot