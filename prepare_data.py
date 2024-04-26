from collections import defaultdict
import numpy as np
import tensorflow as tf

def load_data(digits=None, final_shape=(28, 28), pad_shape=None, threshold=0.5):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Select 2 digits
    if digits != None:
        digit1, digit2 = digits
        train_images, train_labels = select_digits(train_images, train_labels, digit1, digit2)
        test_images, test_labels = select_digits(test_images, test_labels, digit1, digit2)

    # Down sample images
    train_images = train_images[..., np.newaxis] / 255
    test_images = test_images[..., np.newaxis] / 255
    train_images = tf.image.resize(train_images, final_shape).numpy()
    test_images = tf.image.resize(test_images, final_shape).numpy()

    # Change to binary
    if threshold != 0:
        train_images = np.array(train_images > threshold, dtype=np.float32)
        test_images = np.array(test_images > threshold, dtype=np.float32)

    # Remove duplicates
    train_images, train_labels = remove_duplicate(train_images, train_labels)
    test_images, test_labels = remove_duplicate(test_images, test_labels)

    # Pad images
    if pad_shape != None:
        train_images = pad_images(train_images, pad_shape)
        test_images = pad_images(test_images, pad_shape)

    return (train_images, train_labels), (test_images, test_labels)

def select_digits(images, labels, digit1, digit2):
    new_images = []
    new_labels = []
    for image, label in zip(images, labels):
        if label in [digit1, digit2]:
            new_images.append(image)
            if label == digit1:
                new_labels.append(1)
            else:
                new_labels.append(0)
    return np.array(new_images), np.array(new_labels)

def remove_duplicate(images, labels):
    unique_images = defaultdict(int)
    for image in images:
        unique_images[tuple(image.flatten())] += 1

    new_images = []
    new_labels = []
    for image, label in zip(images, labels):
        if unique_images[tuple(image.flatten())] == 1:
            new_images.append(image)
            new_labels.append(label)

    return np.array(new_images), np.array(new_labels)

def pad_images(images, pad_shape):
    pad_shape = (pad_shape[0], pad_shape[1], 1)
    new_images = []
    h, w, c = images[0].shape
    for image in images:
        new_image = np.zeros(pad_shape)
        new_image[:h, :w] = image
        new_images.append(new_image)
    return np.array(new_images)

if __name__=="__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data((28, 28), (32, 32, 1), 0.5)
    print("Train Shape: ", train_images.shape, train_labels.shape)
    print("Test Shape: ", test_images.shape, test_labels.shape)