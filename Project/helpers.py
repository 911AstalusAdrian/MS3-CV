import numpy as np
import tensorflow as tf
import cv2
import os

from glob import glob


train_img_path = r'D:\Uni\MS3-CV\Project\data\train\image'
train_mask_path = r'D:\Uni\MS3-CV\Project\data\train\mask'
test_img_path = r'D:\Uni\MS3-CV\Project\data\test\image'
test_mask_path = r'D:\Uni\MS3-CV\Project\data\test\mask'

def load_images(folder):
    images = []
    for filename in glob(os.path.join(folder, '*.png')):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def resize_images(images, target_size):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return np.array(resized_images)

def prepare_images():
    train_images = load_images(train_img_path)
    train_masks = load_images(train_mask_path)
    test_images = load_images(test_img_path)
    test_masks = load_images(test_mask_path)

    train_images = resize_images(train_images, (256, 256))
    train_masks = resize_images(train_masks, (256, 256))
    test_images = resize_images(test_images, (256, 256))
    test_masks = resize_images(test_masks, (256, 256))

    #normalizing the images
    train_images = train_images / 255.0
    train_masks = train_masks / 255.0
    test_images = test_images / 255.0
    test_masks = test_masks / 255.0

    #adding a channel dimension
    train_images = tf.expand_dims(train_images, axis=-1)
    train_masks = tf.expand_dims(train_masks, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)
    test_masks = tf.expand_dims(test_masks, axis=-1)

    print(f"Train images shape: {train_images.shape}")
    print(f"Train masks shape: {train_masks.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test masks shape: {test_masks.shape}")

    return train_images, train_masks, test_images, test_masks