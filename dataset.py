import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
from numpy.random import normal
import torch
import os
from matplotlib.image import imread
import cv2
from skimage import transform
from markerlab import *

def renderBackground(target, resolution = (1125, 2436)):
    target = cv2.resize(target, dsize=resolution, interpolation=cv2.INTER_CUBIC)
    if np.min(target) < 0:
            target = (target - np.min(target))
    return np.ndarray.astype(target / np.max(target) * 255, int)
def getRandomTransform(img, pattern, max_inclination = np.pi/4):
    α, β, γ = (np.random.rand(3) * 2 - 1) * np.array([max_inclination, max_inclination, np.pi])
    offset = np.random.rand(2) * (np.array([img.shape[1], img.shape[0]]) - np.array([pattern.shape[1], pattern.shape[0]]))
    rotation = np.array([
        [cos(β) * cos(γ), sin(α) * sin(β) * cos(γ) - cos(α) * sin(γ), 0],
        [cos(β) * sin(γ), sin(α) * sin(β) * sin(γ) + cos(α) * cos(γ), 0],
        [0, 0, 1]
    ])
    offset = transform.EuclideanTransform(translation=offset)
    rotation = transform.EuclideanTransform(matrix=rotation)
    shift = transform.EuclideanTransform(translation=-np.array(img.shape[:2]) / 2)
    scale = transform.SimilarityTransform(scale=  0.5/(max(pattern.shape) / max(img.shape)) * (np.random.rand() + 0.7) )
    matrix = np.linalg.inv(shift.params) @ rotation.params @ scale.params @ shift.params @ offset.params
    #matrix = rotation.params @ scale.params @ offset.params
    tform = transform.EuclideanTransform(matrix)
    return tform

def getMarkerDataset(pattern=None, klein = ""):
    maxsize = (4500, 4500, 3)
    if pattern is None: pattern = np.random.randint(low=0, high=255, size=(130, 130, 3))
    Input, Output, binMasked = [], [], []
    cnt = -1
    cnt2 = 0
    for l in os.listdir("Images" + klein):
        cnt += 1
        print(cnt + cnt2)
        img = np.asarray(imread("Images" + klein + "//" + l))
        img = img.__copy__()
        if img.shape[-1] == 4:
            img = img[...,:-1]
        img = renderBackground(img)
        background = np.zeros(shape=img.shape, dtype=int)
        border = np.array([[0, 0, 1], [pattern.shape[0], 0, 1], [0, pattern.shape[1], 1],
                           [pattern.shape[0], pattern.shape[1], 1]]).T
        tform = getRandomTransform(img, pattern)
        new_border = tform.params @ border
        while any(max(new_border[k]) >= img.shape[1-k] or min(new_border[k]) < 0 for k in [0,1]):
            tform = getRandomTransform(img, pattern)
            new_border = tform.params @ border

        background[0:pattern.shape[0], 0:pattern.shape[1]] = pattern
        trans = transform.warp(background, tform.inverse, preserve_range=True, order=1)
        trans = np.ndarray.astype(trans, int)

        marked = np.where(np.greater(trans, 3), trans, img)
        bin_maske = np.where(np.greater(np.amax(trans, axis=2), 0), 1, 0)
        marked = np.astype(marked + np.random.normal(scale = 0, size=marked.shape), int)
        marked = np.clip(marked, a_min=0, a_max=255)
        light = np.zeros(shape=marked.shape, dtype=np.uint8)
        X = tform.params @ np.array([pattern.shape[0]/2, pattern.shape[1]/2, 1])
        light[round(X[1]):round(X[1])+50, round(X[0]):round(X[0])+50] = np.array([255, 255, 0], dtype=np.uint8)

        Output.append(torch.from_numpy(light))
        Input.append(torch.from_numpy(marked))
        binMasked.append(torch.from_numpy(bin_maske))
    torch.save(obj=Input, f="Datasets//Input" + klein)
    torch.save(obj=Output, f="Datasets//Output" + klein)
    torch.save(obj=binMasked, f="Datasets//binary" + klein)

    print("----------------------SAVED------------------------------")

def externDataset(pattern_function=None, klein = False, test_split = 600, repeats = 4, Folder = 'extern_dataset'):
    """
    :param pattern_function: Function that returns pattern to be projected
    :param Folder:
    :param klein: Should Images come from the ImagesKlein Folder
    :param test_split:
    :param repeats: How often should each Image be usef
    :return: populates the folder extern_dataset with a dataset, given the images in the
    "Images" folder and the pattern provided to the function
    """
    klein = 'Klein' if klein else ''
    for f in ['train_images', 'train_masks', 'valid_images', 'valid_masks']:  #delete old dataset if it exists
        import shutil
        from PIL import Image
        folder = Folder + '/' + f
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    fig = None
    cnt2 = 0
    for rep in range(repeats):
        cnt = -1
        for l in os.listdir("Images" + klein):
            pattern = pattern_function()
            cnt += 1
            train_test = 'valid' if cnt > test_split else 'train'
            print("Image Nr: ", cnt)
            img = np.asarray(imread("Images" + klein + "//" + l))
            img = img.__copy__()
            if np.average(img) < 10:
                continue
            if img.shape[-1] == 4:
                img = img[...,:-1]
            if min(img.shape[:-1]) < 500: continue  #Bicubic does not work with extreme upscaling

            border = np.array([[0, 0, 1], [pattern.shape[0], 0, 1], [0, pattern.shape[1], 1],
                               [pattern.shape[0], pattern.shape[1], 1]]).T
            tform = getRandomTransform(img, pattern)
            new_border = tform.params @ border
            while any(max(new_border[k]) >= img.shape[1-k] or min(new_border[k]) < 0 for k in [0,1]):
                tform = getRandomTransform(img, pattern)
                new_border = tform.params @ border

            background = np.zeros(shape=img.shape, dtype=int)
            background[0:pattern.shape[0], 0:pattern.shape[1]] = pattern
            white_background = np.zeros(shape=img.shape, dtype=int)
            white_background[0:pattern.shape[0], 0:pattern.shape[1]] = 255
            marked = transform.warp(background, tform.inverse, preserve_range=True, order=1)
            marked = np.ndarray.astype(marked, np.uint8)
            white_marked = transform.warp(white_background, tform.inverse, preserve_range=True, order=1)

            image = np.where(np.greater(white_marked, 0), marked, img)
            image = np.astype(image, float) + normal(loc=0, scale=0.5, size=image.shape)
            image = np.astype(image, np.uint8)
            white_marked = np.astype(white_marked, np.uint8)
            white_marked = Image.fromarray(white_marked)
            image = Image.fromarray(image)
            image.save(fp=Folder + '/' + train_test + "_images/" + "water_body_" + str(cnt + cnt2) + ".jpg")
            white_marked.save(fp=Folder + '/' + train_test + "_masks/" + "water_body_" + str(cnt + cnt2) + ".jpg")
        cnt2 += cnt


def generateImageWithKnownborders(pattern, image="Images/bike_350.bmp"):
    from models import randomlyProjectPattern
    img = np.asarray(imread(image)) if isinstance(image, str) else image
    pattern, corners = randomlyProjectPattern(pattern=pattern, add_noise=False, RGB=True)
    img = renderBackground(target=img, resolution=pattern.shape[:-1])
    return np.where(np.greater(pattern, 1), pattern, img), corners


if __name__ == '__main__':
    externDataset(pattern_function=getRandomFourierwBorder, Folder="FourierDataset", repeats=8)














