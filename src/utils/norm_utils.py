import os
from sklearn.utils import shuffle
import torch
import random
import numpy as np

def load_dataset(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        data = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) for name in data]
        masks = [os.path.join(path, "masks", name) for name in data]
        edges = [os.path.join(path, "edges", name) for name in data]
        return images, masks, edges

    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/valid.txt"

    train_img, train_mask, train_edge = load_names(path, train_names_path)
    valid_img, valid_mask, valid_edge = load_names(path, valid_names_path)

    return (train_img, train_mask, train_edge), (valid_img, valid_mask, valid_edge)


""" Shuffle the dataset. """
def shuffling(x, y, z):
    x, y, z = shuffle(x, y, z, random_state=42)
    return x, y, z

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


