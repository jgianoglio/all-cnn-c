#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:23:34 2018

@author: Jim Gianoglio
"""

import numpy as np

# np.linalg.svd -- https://piazza.com/class/j9xdaodf6p1443?cid=983

#train_feats = np.load('dataset/train_feats.npy', encoding='bytes')
#train_labels = np.load('dataset/train_labels.npy', encoding='bytes')
#test_feats = np.load('dataset/test_feats.npy', encoding='bytes')



def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    sample_mean = np.mean(x, axis = 1, keepdims=True)
    return(x - sample_mean)
    


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    sample_variance = np.var(x, axis = 1, keepdims=True)
    return scale * x / np.sqrt(bias + sample_variance)


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    feature_mean = np.mean(x, axis = 0, keepdims=True)
    return (x - feature_mean), (xtest - feature_mean)


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    
    #x = fzm_xTrain
    #xtest = fzm_xTest
    #bias = 0.1
    
    n = np.shape(x)[0] #batch size
    m = np.shape(x)[1] #feature size
    svdInput = (x.T.dot(x))/n + np.eye(m) * bias
    U,S,V = np.linalg.svd(svdInput)
    pca = U.dot(np.diag(1. / np.sqrt(S))).dot(U.T)
    zca_xTrain = x.dot(pca)
    
    zca_xTest = xtest.dot(pca)
    
    return zca_xTrain, zca_xTest


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    
    n = np.shape(x)[0]
    nTest = np.shape(xtest)[0]
    
    szm_x = sample_zero_mean(x)
    szm_xTest = sample_zero_mean(xtest)
    
    gcn_x = gcn(szm_x)
    gcn_xTest = gcn(szm_xTest)
    
    fzm_x, fzm_xTest = feature_zero_mean(gcn_x, gcn_xTest)
    
    zca_x, zca_xTest = zca(fzm_x, fzm_xTest)
    
    new_x = np.reshape(zca_x, (n, 3, image_size, image_size))
    new_xTest = np.reshape(zca_xTest, (nTest, 3, image_size, image_size))
    
    return new_x, new_xTest
