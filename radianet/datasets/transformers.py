# pylint: disable=redefined-outer-name

import numpy as np
from albumentations import (
    Flip,
    ElasticTransform,
    Rotate,
    IAAAffine,
    ShiftScaleRotate
)

from ..config import Config


class Transforms():
    def __init__(self, basic=True, elastic_transform=False,
                 shift_scale_rotate=False):
        transforms = []

        if basic:
            transforms.append(Flip(p=0.5))
            transforms.append(
                Rotate(p=0.5, border_mode=Config.BORDER_MODE, limit=45))

        if elastic_transform:
            transforms.append(ElasticTransform(p=0.2))

        if shift_scale_rotate:
            transforms.append(ShiftScaleRotate(p=0.2))

        transforms.append(IAAAffine(p=1, shear=0.2, mode="constant"))
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image=image)['image']
        return np.transpose(image, (2, 0, 1)).astype(np.float32)


if __name__ == '__main__':

    from urllib.request import urlopen

    import numpy as np
    import cv2

    from .utils import transform_and_show

    def download_image(url):
        data = urlopen(url).read()
        data = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    image = download_image(
        'https://d177hi9zlsijyy.cloudfront.net/wp-content/uploads/sites/2/2018/05/11202041/180511105900-atlas-boston-dynamics-robot-running-super-tease.jpg')

    print(image.shape)
    transform = Transforms(basic=True)
    transform_and_show(transform, image)
