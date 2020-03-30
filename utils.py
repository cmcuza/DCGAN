import numpy as np
import scipy.misc
from matplotlib.pyplot import imread


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3,4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images))+1)
    manifold_w = int(np.ceil(np.sqrt(num_images))+1)
    return manifold_h, manifold_w


def inverse_transform(images):
    return (images+1.)/2.


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def get_lr_image(image_paths, input_height=25, input_width=25, grayscale=False):
    lr_img = []

    for image_path in image_paths:
        image = imread(image_path, grayscale)
        if len(image.shape) == 2:
            image = np.stack((image,)*3, -1)

        lr_img.append(scipy.misc.imresize(image, [input_height, input_width]) / 127.5 - 1)

    return lr_img


def get_lr_hd_image(image_paths, input_height=25, input_width=25,
              output_height=200, output_width=200,
              crop=True, grayscale=False):

    lr_img = []
    hd_img = []

    for image_path in image_paths:
        image = imread(image_path, grayscale)

        if not input_width:
            input_width = input_height

        if crop:
            x, y, W, H = 35, 55, 100, 130
            image = image[y:(y + H), x:(x + W), :]

        lr_img.append(scipy.misc.imresize(image, [input_height, input_width])/127.5 - 1)
        hd_img.append(scipy.misc.imresize(image, [output_height, output_width])/127.5 - 1)

    return np.asarray(lr_img), np.asarray(hd_img)


def psnr(img1, img2):
    img1 = np.floor((img1+1)*127.5) if np.max(img1) <= 1.0 else img1
    img2 = np.floor((img2+1)*127.5) if np.max(img2) <= 1.0 else img2
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def save_lr_hd_img(lr, hd):
    save_images(hd, image_manifold_size(hd.shape[0]),
                './samples/test_hd.png')

    resized_lr_img = []

    for img in lr:
        resized_lr_img.append(scipy.misc.imresize(img, [hd.shape[1], hd.shape[2]]))

    resized_lr_img = np.asarray(resized_lr_img)

    save_images(resized_lr_img, image_manifold_size(resized_lr_img.shape[0]),
                './samples/test_lr.png')