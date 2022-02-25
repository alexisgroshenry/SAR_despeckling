import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import special
import matplotlib.pyplot as plt
from glob import glob
from GenerateDataset import GenerateDataset
from tiilab import *


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle




def normalize_sar(im):
    return ((np.log(im+1e-12)-m)/(M-m)).astype(np.float32)

def denormalize_sar(im):
    return np.exp((np.clip(np.squeeze(im),0,1))*(M-m)+m)




def BCrossEntropy(yHat, y):
    return np.where(y == 1, -tf.log(yHat), -tf.log(1 - yHat))



def load_train_data():
    
    datasetdir = './data/training/'
    
    #name_pile = ['lely1', 'lely2', 'lely3', 'limagne1', 'limagne2', 'marais12', 'marais13'
    #name_pile = ['lely1', 'marais12', 'marais13'] 
    name_pile = ['lely', 'marais1']

    dataset_train = []
    
    for name_p in name_pile:
        test = glob(datasetdir+name_p+'*.npy')
        print(test)
        test.sort()
        im_0 = np.load(test[0])
        im = np.zeros((im_0.shape[0], im_0.shape[1], len(test)))
        for i in range(len(test)):

            im[:,:,i] = normalize_sar(np.load(test[i]))


        dataset_train.append((name_p, im))


    real_data = np.array(dataset_train)

    
    return real_data










def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = normalize_sar(np.load(filelist))
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = normalize_sar(np.load(file))
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1))
    return data





def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))




def save_sar_images(denoised, noisy, imagename, save_dir, groundtruth=None):
    choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22,
           'risoul':306.94, 'limagne':178.43, 'saintgervais':760, 'Serreponcon': 450.0,
          'Sendai':600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
          'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    if groundtruth:
        groundtruthfilename = save_dir+"/groundtruth_"+imagename
        np.save(groundtruthfilename,groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    noisyfilename = save_dir + "/noisy_" + imagename
    np.save(noisyfilename, noisy)
    store_data_and_plot(noisy, threshold, noisyfilename)



def save_map(bm, bm_y, imagename, save_dir, groundtruth=None):

    bm_yname = save_dir + "/bm_y_" + imagename
    im = Image.fromarray(bm_y.astype('float64')).convert('L')
    np.save(bm_yname, bm_y*255)
    im.save(bm_yname.replace('npy','png'))

    bmname = save_dir + "/bm_" + imagename
    im = Image.fromarray(bm.astype('float64')).convert('L')
    np.save(bmname, bm*255)
    im.save(bmname.replace('npy','png'))

def save_mapbm(bm, imagename, save_dir, groundtruth=None):
    #bm = bm*255
    bmname = save_dir + "/_" + imagename
    im = Image.fromarray(bm.astype('float64')).convert('L')
    np.save(bmname, bm)
    im.save(bmname.replace('npy','png'))

