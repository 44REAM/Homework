import numpy as np
import pydicom
import os

def load_slices(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path) if s.endswith('dcm')]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_slices_hounsfield(slices):
    images = np.stack([s.pixel_array for s in slices])
    images = images.astype(np.int16)

    images[images == -2000] = 0
    
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    
    if slope != 1:
        images = slope * images.astype(np.float64)
        images = images.astype(np.int16)

    images += np.int16(intercept)

    return np.array(images, dtype=np.int16)

