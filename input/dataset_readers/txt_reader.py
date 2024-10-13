"""

Script name: "input/dataset_readers/txt_reader.py"\n
Goal of the script: Contains the function that converts a txt format dataset into a python list.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""

def read_prompts_from_file(file_path):
    """
    Turns a file at file_path* into a list of strings\n
    :param file_path: *the path to our file
    :return: The list of strings
    """
    with open(file_path, 'r') as file:
        prompts = [line.strip() for line in file.readlines() if line.strip()]
    return prompts