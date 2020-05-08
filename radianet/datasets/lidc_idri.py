import pandas as pd
import numpy as np
from .utils import split_data

class LIDCDataset():
    def __init__(self, n_sample = 100, channels = 1, width = 64, height = 64, deep = 12):
        self.data = np.random.randn(n_sample, channels, width, height, deep).astype(np.float32)
        self.label = np.array([np.random.randint(0, 2) for x in range(n_sample) ]).astype(np.float32)


    def get_data(self, idx):
        idx = int(idx)
        return self.data[idx]

def get_lidc_label(path, label_type = None):
    # path to csv file
    df = pd.read_csv(path)

    df = df[df.Diagnosis != 0]
    
    if label_type == 'malignant':

        df[df.Diagnosis == 1] = 0
        df[df.Diagnosis != 0] = 1
    elif label_type == 'source':
        df = df[df.Diagnosis != 1]
        df[df.Diagnosis == 2] = 0
        df[df.Diagnosis == 3] = 1
        
    ID = df.id.to_numpy()
    label = df.Diagnosis.to_numpy()

    partition, label = split_data(ID, label)
    return partition, label


if __name__ == "__main__":
    path = "D:\\GoogleDrive\\dataset\\radiology\\TCIA_LIDC-IDRI\\tcia-diagnosis-data-2012-04-20.csv"
    partition, label = get_lidc_label(path)
    print(label)
