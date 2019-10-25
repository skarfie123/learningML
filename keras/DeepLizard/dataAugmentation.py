import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10, horizontal_flip=True)

image_path = "path/tp/image.jpg"
image = np.expand_dims(ndimage.imread(image_path), 0)
aug_iter = gen.flow(image)
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)] # generate 10 images