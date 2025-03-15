"""

Script name: "input/embedding/preprocess.py"\n
Goal of the script: Contains functions used to preprocess a single word.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""


# Libraries
import string

def rid_of_i_we(s):
    """
    Gets a lowercase string s, that has no punctuation, and rids of the words "i" "we" and "me", replacing them with the word "user".\n
    """
    new = ""
    for w in s.split():
        if w == 'i' or w == 'we' or w =='me':
            new += ' user'
        else:
            new += " " + w
    return new[1:]

def rid_of_punctuation(s):
    """
    Gets a lowercase string s, and rids of any punctuation.
    """
    new = ""
    for w in s.split():
        for l in range(len(w)):
            if not w[l] in string.punctuation:
                new += w[l]
        new += " "

    return new[:-1]

def rid_of_ing(s):
    """
    Gets a lowercase string s, that has no punctuation or the words "i" "we" or "me", and rids of any "ing" endings. NOTE: Doesn't work if the word in its base form ends with e. Examples:\n
    fighting -> fight.\n
    putting -> put.\n
    waving -> wav instead of wave.\n
    """
    new = ""
    v = ['a', 'e', 'i', 'o', 'u']
    for w in s.split():
        if w[-3:] == "ing":
            old = new
            new += w[:-3] + " "
            if len(new) > 4 and not new[-3] in v and not new[-5] in v and new[-4] in v:
                new = old
                new += w[:-4] + " "
        else:
            new += w + " "
    return new[:-1]

def rid_of_preposition(s):
    """
    Gets a lowercase string s, that has no punctuation or the words "i" "we" or "me" or ing in the end, rids of any prepositions.\n
    """
    new = ""
    for w in s.split():
        if not w in ["a", "an", "some", "is", "are", "am", "the"]:
            new += w + " "
    return new[:-1]

# Feel free to add more functions as needed!