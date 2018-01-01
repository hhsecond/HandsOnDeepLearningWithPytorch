import os

from scipy import misc
import numpy as np


def balckandwhite_pedestrians(img_path):
    img = misc.imread(img_path)
    pedestrian_rgb = np.array([64, 64, 0]).reshape(1, 1, 3)
    grayscale = np.all(img == pedestrian_rgb, 2)
    misc.imsave(img_path, grayscale)


if __name__ == '__main__':
    counter = 0
    path = '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid'
    for folder in os.listdir(path):
        label_folder = os.path.join(path, folder, 'labels')
        for image in os.listdir(label_folder):
            counter += 1
            img_path = os.path.join(label_folder, image)
            balckandwhite_pedestrians(img_path)
            print(counter)
