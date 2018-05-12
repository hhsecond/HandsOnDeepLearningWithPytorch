# Utils has the utilites I have used outside the models
import os

from scipy import misc
import numpy as np


def balckandwhite_pedestrians(img_path):
    img = misc.imread(img_path)
    pedestrian_rgb = np.array([64, 64, 0]).reshape(1, 1, 3)
    out = np.all(img == pedestrian_rgb, 2)
    print(out.shape)
    misc.imsave(img_path, out)


if __name__ == '__main__':
    counter = 0
    path = '/home/hhsecond/mypro/ThePyTorchBook/HandsOnDeepLearningWithPytorch/camvid/'
    for folder in os.listdir(path):
        for image in os.listdir(os.path.join(path, folder, 'labels')):
            counter += 1
            img_path = os.path.join(path, folder, 'labels', image)
            balckandwhite_pedestrians(img_path)
