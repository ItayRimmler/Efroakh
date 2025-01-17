
# Efroakh

**Efroakh** is a self-programming video game, utilizing neural networks to adapt its gameplay based on player input prompt.

Currently developing v.2. Efroakh v.1 is the program in its first 12 commits

## Features

- Neural network-driven gameplay adaptation
- Dynamic interaction with player actions
- Customizable hyperparameters for fine-tuning the game’s AI
- **In the future: an actual game :P**

## Installation

**Currently under work. Stay tuned!**

## Usage

Run the game:
```bash
python main.py
```

## Contributing

Feel free to submit issues and pull requests! If you have future ideas of things to add to the game that are analyzed by prompts, feel free to contribute ideas!

## Versions (Branches)

- CBOW: Here I attempted to classify the strings using CBOW Word2Vec.
- Sentence-encoding: Here I attempted to encode and label whole sentences.

## FAQ


**Q** - Why so many branches?

**A** - I do this project and learning in the process. So I sometimes need to fundementally change the code, but I don't want to get rid of it.


**Q** - Why did you not continue with CBOW branch?

**A** - I didn't think this through enough, and I lacked experience in NLP, so I initally believed that CBOW will help me classify sentences. Apperantly, I needed to convert the sentences into vectors somehow, then label each sentence, then work with this.


**Q** - Why leaving CBOW branch then, if it doesn't work necessarily?

**A** - Umm... I really like what I did there in general, even if it's useless for the project. It stays mostly for the ego and because I wanna flex. So there ya go.


**Q** - What's the purpose of the Sentence-encoding branch?

**A** - Attempt what CBOW branch had tried, only this time convert the sentences into vectors somehow, then label each sentence, then work with this.


**Q** - Do you plan to add more to that branch?

**A** - Not necessarily. I do have more plans. I do this while simultaniously learning RNN and more subjects in NLP. So the plan is to try this, then in the same branch\in a different one switch my feedforward NN with RNN and apply more advanced techniques in NLP
