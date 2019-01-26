import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_classes():
    """ Load class names """
    return np.load('lib/classes.npy')

def get_label_encoder_y():
    """ Get preloaded label encoder """
    labelEncoder = LabelEncoder()
    labelEncoder.classes_ = np.load('lib/classes.npy')
    return labelEncoder

def read_dataset_file(file_name, number_of_items=None, shuffle=True, frac=1):
    """ Read dataset file and get some rows """
    dataset = pd.read_csv(file_name)
    if shuffle:
        dataset = dataset.sample(frac=frac)
    if number_of_items is not None:
        dataset = dataset[:number_of_items]
    return dataset

def read_dataset_folder(path, number_of_items=None, frac=1):
    """ Read the entire dataset folder """
    class_names = get_classes()
    file_contents = []
    for file in class_names:
        file_contents.append(read_dataset_file(path + file + '.csv', number_of_items=number_of_items, frac=frac))
    dataset = pd.concat(file_contents).sample(frac=1)
    del file_contents
    return dataset
