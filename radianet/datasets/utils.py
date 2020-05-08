
from sklearn.model_selection import train_test_split
import numpy as np

def split_data(x,y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)
    partition = {
        'train':x_train,
        'val' : x_val
    }
    label = {
        'y_train': y_train,
        'y_val':y_val
    }
    return partition, label