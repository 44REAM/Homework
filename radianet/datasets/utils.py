from matplotlib import pyplot as plt

def transform_and_show(transform, image):
    image = transform(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image[1,:,:])
    plt.show()

def show(image):

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()