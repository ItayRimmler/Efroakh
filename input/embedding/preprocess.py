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


lowercase = lambda a: [w.lower() for w in a]
rid_of_i_we = lambda a: [w if not w == "i" and not w == "we" and not w == "me" else "user" for w in a ]
def rid_of_punctuation(a):
    for w in a:
        if w in string.punctuation:
            a.remove(w)
        else:
            for l in w:
                if l in string.punctuation:
                    w.replace(l, "")
                    break
    return a
def rid_of_ing(a):
    for i in range(len(a)):
        if "ing" == a[i][-3:]:
            a[i] = a[i][:-3]
    return a
def rid_of_preposition(a):
    for w in a:
        if w in ["a", "an", "some", "is", "are", "am"]:
            a.remove(w)
    return a
# Feel free to add more functions as needed!