# GANS

## Introduction
- The goal of this project is to create images of faces using generative adversarial networks.
- Gans was first developed by Ian Goodfellow in 2014.
- This is a combination of supervised and unsupervised algorithm.
- GANS contains two parts Generator and Discriminator.
- Noise is given as input to a series of convolutional layers which interm generates images with some noise.(Generator network)
- Image celeba dataset form kaggle is taken as input for discriminator network.
- The images generated by generator network are compared with the outcome of discriminator network using sigmoid cross entropy function. If the probability of the image is closer to 1 that means the image is considered to be real or else it is fake image.
- For every batch the generator tries to fool the discriminator by making the generator o/p closer to real images. 

## Software
- Tensorflow 2.2
- python 2.7

## Instructions
- Download the code from the repository.
- Run the file "python practice2.py".
