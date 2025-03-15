"""

Script name: "assets\asset_creator.py"\n
Goal of the script: Create a dataset automatically.\n
Part of project: "Efroakh"\n
Description of project: A video game that programs itself.\n
Ways to contact me if something went wrong in the code: itay.rimmler@gmail.com\n
Uploaded to GitHub in the link: https://github.com/ItayRimmler?tab=repositories\n
Deez: Nuts\n

"""

import random
from hyperparameters import *

# Directions for labels
directions = ["up", "down", "left", "right"]

# Words to generate diverse prompts, can be edited further
verbs = ["place", "set", "move", "position", "make", "shift", "adjust", "put", "head", "direct",
         "take", "drag", "flip", "bring", "push", "pull", "drop", "align", "point", "guide"]

objects = ["goal", "target", "finish line", "objective", "destination", "marker", "spot", "checkpoint",
           "flag", "area", "zone", "circle", "square", "point", "node", "edge", "place", "corner", "line"]

prepositions = ["towards", "to", "on", "at", "in", "below", "above", "toward", "heading", "near",
                "around", "over", "under", "along", "against", "beside", "within", "behind", "past"]

locations = ["top", "bottom", "left", "right", "north", "south", "up", "down", "the side", "the right side",
             "the left side", "the peak", "the base", "the edge", "the corner", "the ceiling", "the floor",
             "high", "low", "the middle"]

# Generate a dataset of 100,000 unique prompts and labels
num_prompts = DATASET_SIZE

prompts = []
labels = []

# Function to create a short sentence
def create_short_sentence():
    return random.choice(["up", "down", "left", "right", "above", "below", "top please", "bottom please"])

# Function to create a medium sentence
def create_medium_sentence():
    verb = random.choice(verbs)
    loc = random.choice(["up", "down", "left", "right", "above", "below", "top", "bottom"])
    obj = random.choice(objects)
    return random.choice([
        f"{verb.capitalize()} it {loc}",
        f"{obj.capitalize()} {loc} please",
        f"{verb.capitalize()} {obj} {loc}"
    ])

# Create prompts with varying sentence structures
for _ in range(num_prompts):
    sentence_type = random.choices(["short", "medium", "long"], weights=[0.1, 0.2, 0.7])[0]

    if sentence_type == "short":
        prompt = create_short_sentence()
    elif sentence_type == "medium":
        prompt = create_medium_sentence()
    else:  # Long sentence
        verb = random.choice(verbs)
        obj = random.choice(objects)
        prep = random.choice(prepositions)
        loc = random.choice(locations)
        prompt = f"{verb.capitalize()} the {obj} {prep} {loc}."

    # Map the location to the correct label
    if "up" in prompt or "top" in prompt or "north" in prompt:
        label = "up"
    elif "down" in prompt or "bottom" in prompt or "south" in prompt:
        label = "down"
    elif "left" in prompt:
        label = "left"
    elif "right" in prompt:
        label = "right"
    else:
        label = random.choice(directions)  # If somehow the location is unexpected, default to random

    prompts.append(prompt)
    labels.append(label)

# Save the prompts and labels to respective text files
prompts_file_path = 'prompts.txt'
labels_file_path = 'labels.txt'

# Write to file1 (prompts)
with open(prompts_file_path, 'w') as prompts_file:
    prompts_file.write("\n".join(prompts))

# Write to file2 (labels)
with open(labels_file_path, 'w') as labels_file:
    labels_file.write("\n".join(labels))
