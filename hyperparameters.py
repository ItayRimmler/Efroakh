HIDDEN_LAYER_SIZES = [64, 128, 64]
EPOCHS = 1000
LR = 0.01
CLIP = 1.5
LEAK = 0.5
TRAIN = lambda s: int(0.8 * s)
TEST = lambda s: int(0.2 * s)