import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PIL import Image, ImageOps
import time
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
from resizeimage import resizeimage

# print(tf.__version__)
tf.compat.v1.disable_eager_execution()





tf.compat.v1.set_random_seed(1)

celeba_img_dir = "/home/akarsh/Documents/yash/dataset/"


im_height = 64
im_width = 64


celeba_im_height = 64
celeba_im_width = 64
celeba_im_channels = 1




def get_celeba_dataset(batch_size = 100, option = 0):
    files = os.listdir(celeba_img_dir)
    # print('files:',files)
    files = sorted(files)
    print("total number of images:",len(files))
    # true_imgs = np.zeros(shape=[len(files),celeba_im_height, celeba_im_width,celeba_im_channels], dtype = np.float32)
    true_imgs = np.zeros(shape=[2000,celeba_im_height, celeba_im_width,celeba_im_channels], dtype = np.float32)
    # print('true_imgs:',true_imgs)
    
    i=0
    for file in files:
        # loading images form directory
        im = Image.open(celeba_img_dir+ '/'+file)
        im = ImageOps.grayscale(im)
        cover = resizeimage.resize_cover(im, [64, 64])
        img = np.array(cover, dtype = np.float32)
        # print(img.shape)
        temp = 2* img/255.0-1
        # print(temp.shape)
        # normalization_step
        true_imgs[i,:,:,0] = 2* img/255.0-1
        i = i+1
        if i>=2000:
            break
        # print('i:',i)

    # Shuffling the images
    np.random.shuffle(true_imgs)

    return true_imgs



def conv_bn_leaky_relu(scope_name, input, filter, k_size, stride=(1,1), padd = 'SAME'):
    with tf.compat.v1.variable_scope(scope_name,reuse = tf.compat.v1.AUTO_REUSE) as scope:
        conv = tf.compat.v1.layers.conv2d(inputs = input, filters = filter,kernel_size = k_size,strides = stride,padding = padd)
        batch_norm = tf.compat.v1.layers.batch_normalization(inputs = conv, training = True)
        a = tf.nn.leaky_relu(batch_norm, name = scope.name)

    return a


def transpose_conv_bn_relu(scope_name, input, filter ,k_size, stride=(1,1), padd = 'VALID'):
    with tf.compat.v1.variable_scope(scope_name, reuse = tf.compat.v1.AUTO_REUSE) as scope:
        tr_conv = tf.compat.v1.layers.conv2d_transpose(input,filter,k_size,stride, padd, activation = None)
        batch_norm = tf.compat.v1.layers.batch_normalization(inputs = tr_conv, training = True)
        a = tf.nn.relu(batch_norm, name = scope.name)

    return a



def safe_mkdir(path):

    try:
        os.mkdir(path)
    except OSError:
        pass