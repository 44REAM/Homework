
import pandas as pd
from pylidc.utils import consensus
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pylidc as pl


class LIDCDataset():
    def __init__(self, n_sample=100, channels=1, width=64, height=64, deep=12):
        self.data = np.random.randn(
            n_sample, channels, width, height, deep).astype(np.float32)
        self.label = np.array([np.random.randint(0, 2)
                               for x in range(n_sample)]).astype(np.float32)

    def get_data(self, idx):
        idx = int(idx)
        return self.data[idx]


def get_lidc_label(path, label_type='malignant'):
    # path to csv file
    cutout = ['LIDC-IDRI-0100', 'LIDC-IDRI-0118', 'LIDC-IDRI-0124']
    df = pd.read_csv(path)

    for cut in cutout:
        df = df[df.id != cut]

    df = df[df.Diagnosis != 0]

    if label_type == 'malignant':

        df['Diagnosis'][df.Diagnosis == 1] = 0
        df['Diagnosis'][df.Diagnosis != 0] = 1
    elif label_type == 'source':
        df = df[df.Diagnosis != 1]
        df['Diagnosis'][df.Diagnosis == 2] = 0
        df['Diagnosis'][df.Diagnosis == 3] = 1

    partition = df.id.to_numpy()
    label = df.Diagnosis.to_numpy()

    return partition, label


def extraction(path, label_type='malignant', debug=False, show=False):
    partition, label = get_lidc_label(path, label_type)

    for name in partition:
        scan, vol = get_scan_vol(name)
        nodules = get_nodules(scan)

        for nodule in nodules:
            image = get_image(nodule, vol)
            image = preprocess_image(image)
            if show:
                print(image.shape)
                visualize(image)

        if debug:
            break


def visualize(image):
    plt.imshow(image)
    plt.show()


def preprocess_image(image):
    dim = (240, 240)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = cv2.merge((image,image,image))
    return image


def get_scan_vol(name):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == name).first()
    vol = scan.to_volume()
    return scan, vol


def get_nodules(scan):

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    n_nod = len(nods)
    print("%s has %d nodules." % (scan, n_nod))
    return nods


def get_image(nodule, vol):

    anns = nodule

    print(anns[0].malignancy, anns[0].Malignancy)

    _, cbbox, _ = consensus(anns, clevel=0.5,
                            pad=[(120, 120), (120, 120), (0, 0)])

    # Get the central slice of the computed bounding box.
    k = int(0.5*(cbbox[2].stop - cbbox[2].start))

    return vol[cbbox][:, :, k]
    # ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)

    # # Plot the annotation contours for the kth slice.
    # colors = ['r', 'g', 'b', 'y']
    # for j in range(len(masks)):
    #     for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
    #         label = "Annotation %d" % (j+1)
    #         plt.plot(c[:,1], c[:,0], colors[j], label=label)

    # # Plot the 50% consensus contour for the kth slice.
    # for c in find_contours(cmask[:,:,k].astype(float), 0.5):
    #     plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')
    # ax.axis('off')
    # ax.legend()
    # plt.tight_layout()
    # #plt.savefig("../images/consensus.png", bbox_inches="tight")
    # plt.show()
if __name__ == "__main__":
    path = "D:\\GoogleDrive\\dataset\\radiology\\TCIA_LIDC-IDRI\\tcia-diagnosis-data-2012-04-20.csv"
    extraction(path, show=True, debug=False)
