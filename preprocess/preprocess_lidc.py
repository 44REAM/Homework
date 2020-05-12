import os

import pandas as pd
from pylidc.utils import consensus
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom

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

def extraction(path, label_type='malignant', debug=False,
               show=False, save=False, save_path=None):
    partition, label = get_lidc_label(path, label_type)

    savefile = {
        'nodule_name': [],
        'nodule_label': []
    }
    count = 0
    for diagnosis, name in zip(label, partition):
        scan = get_scan(name)
        vol = get_volume(scan)
        nodules = get_nodules(scan)

        for nodule in nodules:
            image = get_image(nodule, vol)
            image = preprocess_image(image)

            if show:
                visualize(image)
            if save:
                name = str(count)

                savefile['nodule_name'].append(name)
                savefile['nodule_label'].append(diagnosis)

                save_name = save_path + name
                np.save(save_name, image)
                count -= -1

        if debug:
            break

    df = pd.DataFrame.from_dict(savefile)
    csv_name = save_path + "df.csv"
    df.to_csv(csv_name)


def get_slices(path, sorted_names):
    sorted_names = sorted_names.split(",")

    slices = []
    for s in sorted_names:
        print(s)
        if len(s) == 6:
            name = path + '/1-0' + s
        elif len(s) == 5:
            name = path + '/1-00' + s
        else:
            name = path + '/1-' + s
        slices.append(pydicom.dcmread(name))
        break

    return slices


def get_pixels_hu(slices, scan):

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0

    image = scan.to_volume()
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    image = np.array(image, dtype=np.int16)

    return image


def get_volume(scan):
    # path = scan.get_path_to_dicom_files()
    # sorts_names = scan.sorted_dicom_file_names

    # slices = get_slices(path, sorts_names)
    # images = get_pixels_hu(slices, scan)

    # plt.imshow(images[:,:,150], cmap=plt.cm.bone)
    # plt.show()
    images = scan.to_volume()
    min_pixel = np.min(images)
    images[images == min_pixel] = -1024

    return images


def visualize(image):
    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    plt.imshow(image, cmap=plt.cm.gray)

    plt.show()


def preprocess_image(image):
    dim = (224, 224)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = cv2.merge((image, image, image))

    return image


def get_scan(name):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == name).first()

    return scan


def get_nodules(scan):

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    n_nod = len(nods)
    print("%s has %d nodules." % (scan, n_nod))
    return nods


def normalize(image):

    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def zero_conter(image):
    PIXEL_MEAN = 0.25

    image = image - PIXEL_MEAN
    return image


def get_image(nodule, vol):

    anns = nodule

    print(anns[0].malignancy, anns[0].Malignancy)

    _, cbbox, _ = consensus(anns, clevel=0.5,
                            pad=[(120, 120), (120, 120), (0, 0)])

    # Get the central slice of the computed bounding box.
    k = int(0.5*(cbbox[2].stop - cbbox[2].start))
    image = normalize(vol[cbbox][:, :, k])
    image = zero_conter(image)

    return image
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

    save_path = "D:\\GoogleDrive\\dataset\\radiology\\TCIA_LIDC-IDRI\\preprocess\\"
    extraction(path, show=False, debug=False, save=True, save_path=save_path)
