"""

Script name: "input/embedding/preprocess.py"\n
Goal of the script: Contains functions used to preprocess a split corpus of words.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Libraries
import string


lowercase = lambda w: w.lower()
rid_of_i_we = lambda w: w if not w == "i" and not w == "we" and not w == "me" else "user"
def rid_of_punctuation(w):
    if w in string.punctuation:
        return ""
    else:
        for l in w:
            if l in string.punctuation:
                w.replace(l, "")
    return w
def rid_of_ing(w):
    if w[-3:] == "ing":
        return w[:-3]
    return w
def rid_of_preposition(w):
    if w in ["a", "an", "some", "is", "are", "am"]:
            return ""
    return w
# Feel free to add more functions as needed!