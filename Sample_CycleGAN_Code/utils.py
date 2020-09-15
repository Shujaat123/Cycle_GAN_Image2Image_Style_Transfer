import os
# import tensorflow as tf
# import re
import numpy as np
# import copy
import PIL.Image as Image  # , ImageEnhance
# from scipy.ndimage import rotate
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image

def get_img_path(target_dir):
    input_img_files, target_img_files = data_files(target_dir)
    return input_img_files, target_img_files

def data_files(data_path):
    input_img_dir = os.path.join(data_path, "input_das")
    # input_img_dir = os.path.join(data_path, "input_das_PWsALL")
    # input_img_dir = os.path.join(data_path, "input_cardiac_MLA")
    # input_img_dir = os.path.join(data_path, "input_das_PWs3")
    # input_img_dir = os.path.join(data_path, "input_das_PWs11")
    # input_img_dir = os.path.join(data_path, "input_das_mixed")
    # target_img_dir = os.path.join(data_path, "target_beta_20_H_4_cardiac_MLA")
    target_img_dir = os.path.join(data_path, "label_deepDeconv")
    # target_img_dir = os.path.join(data_path, "label_das_despeckle")
    # target_img_dir = os.path.join(data_path, "label_deconv_despeckle")
    # target_img_dir = os.path.join(data_path, "label_deconv_despeckle_mixed")

    input_img_files = all_files_under(input_img_dir, extension=".bmp")
    target_img_files = all_files_under(target_img_dir, extension=".bmp")

    return input_img_files, target_img_files


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]


    if sort:
        filenames = sorted(filenames)
    return filenames


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def get_train_batch(train_input_img_files, train_target_img_files, num_train, batch_size, Training = True):
    train_indices = np.random.choice(num_train, batch_size, replace=False)
    if (Training):
        train_indices2 = np.random.choice(num_train, batch_size, replace=False)

    batch_size = len(train_indices)
    batch_input_img_files, batch_target_img_files = [], []
    if (Training):
        for ind, idx in enumerate(train_indices):
            batch_input_img_files.append(train_input_img_files[idx])
            if (Training):
                batch_target_img_files.append(train_target_img_files[train_indices2[ind]])
            else:
                batch_target_img_files.append(train_target_img_files[idx])
    else:
        for ind, idx in enumerate(train_indices):
            batch_input_img_files.append(train_input_img_files[ind])
            batch_target_img_files.append(train_target_img_files[ind])

    # load images
    input_images = image_files2arrs_normalize(batch_input_img_files)
    target_images = image_files2arrs_normalize(batch_target_img_files)
    # assert (np.min(vessel) == 0 and np.max(vessel) == 1)

    all_input_images, all_target_images = [], []

    for i in range(batch_size):
        all_input_images.append(input_images[i])
        all_target_images.append(target_images[i])


    input_images = np.asarray(all_input_images)
    target_images = np.asarray(all_target_images)

    return input_images, target_images


def image_files2arrs_normalize(filenames):
    # img_shape = image_shape(filenames[0])
    # images_arr = None
    target_size = (256, 256)
    # if len(img_shape) == 3:
    #     images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    # elif len(img_shape) == 2:
    images_arr = np.zeros((len(filenames), 256, 256), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        img = img.resize(target_size, Image.ANTIALIAS)
        images_arr[file_index] = np.asarray(img).astype(np.float32)/255
        # images_arr[file_index] = np.asarray(img).astype(np.float32)
        # min_array = np.min(images_arr[file_index])
        # max_array = np.max(images_arr[file_index])
        # images_arr[file_index] = normalize(images_arr[file_index], min_array, max_array)
        # THIS IS MY LINE
        # images_arr[file_index]= image.extract_patches_2d(images_arr[file_index], patch_size=(256, 256), max_patches=1)
        # images_arr[file_index] = images_arr[file_index] - 1
    return images_arr


def image_files2arrs_normalize_(filenames):
    # img_shape = image_shape(filenames[0])
    # images_arr = None
    target_size = (256, 256)
    # if len(img_shape) == 3:
    #     images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    # elif len(img_shape) == 2:
    images_arr = np.zeros((len(filenames), 256, 256), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        img = img.resize(target_size, Image.ANTIALIAS)
        images_arr[file_index] = np.asarray(img).astype(np.float32)/255
        # images_arr[file_index] = np.asarray(img).astype(np.float32)
        # images_arr[file_index] = images_arr[file_index] / 127.5
        # images_arr[file_index] = images_arr[file_index] - 1
    return images_arr


def normalize(array, min_array, max_array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j] = (array[i][j] - min_array)/(max_array - min_array)
    return array


def normalize_(array, min_array, max_array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                array[i][j][k] = (array[i][j][k] - min_array)/(max_array - min_array)
    return array



def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape



def train_next_batch(batch_size, num_train, train_img, train_vessel, Training = True):
    # train_indices = np.random.choice(num_train, batch_size, replace=False)
    # train_imgs, train_vessels = get_train_batch(train_img, train_vessel, train_indices.astype(np.int32))
    train_imgs, train_vessels = get_train_batch(train_img, train_vessel, num_train, batch_size, Training)
    train_vessels = np.expand_dims(train_vessels, axis=3)
    train_imgs = np.expand_dims(train_imgs, axis=3)
    return train_imgs, train_vessels


def val_next_batch(val_img, val_vessel, i):
    val_indices = i
    val_imgs, val_img_, val_vessels = get_val_batch(val_img, val_vessel, val_indices)
    val_vessels = np.expand_dims(val_vessels, axis=3)
    val_imgs = np.expand_dims(val_imgs, axis=3)
    val_img_ = np.expand_dims(val_img_, axis=3)
    return val_imgs, val_img_, val_vessels


def get_val_batch(val_input_img_files, val_target_img_files,  val_indices):
    batch_size = 1
    batch_input_img_files, batch_target_img_files = [], []
    batch_input_img_files.append(val_input_img_files[val_indices])
    batch_target_img_files.append(val_target_img_files[val_indices])

    # load images
    input_images = image_files2arrs_normalize(batch_input_img_files)
    image_ = image_files2arrs_normalize_(batch_input_img_files)
    target_images = image_files2arrs_normalize(batch_target_img_files)

    all_input_images, all_image_, all_target_images = [], [], []

    for i in range(batch_size):
        all_input_images.append(input_images[i])
        all_image_.append(image_[i])
        all_target_images.append(target_images[i])

    input_images = np.asarray(all_input_images)
    image_ = np.asarray(all_image_)
    target_images = np.asarray(all_target_images)

    return target_images, image_, np.round(target_images)


def im_write(image, path):
    if len(image.shape) == 2:
        image = image
    elif len(image.shape) == 4:
        for i in range(image.shape[0]):
            image = image[i]
            image = np.squeeze(image, axis=2)
    else:
        image = np.squeeze(image, axis=2)
    min_i = np.min(image)
    max_i = np.max(image)
    if min_i == max_i:
        return plt.imsave(path, image, cmap='gray')
    else:
        return io.imsave(path, image)

