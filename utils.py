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



def load_train_data(load_all=True):
    
    datasetdir = './data/training/'
    # get the name of the piles for training (files must follow name convention "pilename_blabla.npy") 
    filelist = glob(datasetdir+'*.npy') 
    name_pile = list(set([file.replace(datasetdir,'').split('_')[0] for file in filelist])) 
    name_pile.sort() 
    #name_pile = ['lely1', 'lely2', 'lely3', 'limagne1', 'limagne2', 'marais12', 'marais13'
    #name_pile = ['lely1', 'marais12', 'marais13'] 
    # name_pile = ['lely', 'marais1']
    dataset_train = []
    for name_p in name_pile:
        test = glob(datasetdir+name_p+'*.npy')
        print(test)
        test.sort()
        if load_all :
            im_0 = np.load(test[0])
            im = np.zeros((im_0.shape[0], im_0.shape[1], len(test)))
            for i in range(len(test)):
                im[:,:,i] = normalize_sar(np.load(test[i]))
            dataset_train.append((name_p, im))

        else :
            im = []
            for i in range(len(test)):
                im.append(test[i])

            image = np.load(test[0])
            im_h = np.size(image,0)
            im_w = np.size(image,1)
            dataset_train.append((name_p, im,im_h,im_w))
    real_data = np.array(dataset_train)
    return real_data


def load_sar_images(datasetdir, pile, mode='default'):
    """
        Mode :
            - default : only in the order (0-1 2-3 3-4 5-6)
            - reverse : the same as default plus the reversed one (0-1 1-0 2-3 3-2 etc)
            - all : only when pile is 2 : (0-1 0-2 0-3 0-4 etc...)
    """
    # get the name of the piles for evaluation (files must follow name convention "pilename_blabla.npy")
    filelist = glob(datasetdir+'*.npy')
    name_pile = list(set([file.replace(datasetdir,'').split('_')[0] for file in filelist]))
    name_pile.sort()
    data = []
    eval_files = []
    for name_p in name_pile:
        files_p = glob(datasetdir+name_p+'*.npy')
        files_p.sort()
        # print("Pile : {} / Len : {}".format(name_p,len(files_p)))
        assert pile <= len(files_p), "Not enough images for the pile selected in {}".format(datasetdir+name_p)
        # for now we build a single pile 
        im_ref = np.load(files_p[0])
        if mode == 'default' :
            """
                0-1 2-3 4-5 6-7 etc
            """
            for k in range(len(files_p) // pile) : # number of subpile we can find
                im = np.zeros((1,im_ref.shape[0], im_ref.shape[1], pile))
                for idx,i in enumerate(range(k*pile, k*pile +pile)):
                    im[0,:,:,idx] = normalize_sar(np.load(files_p[i]))

                # add the pile images and the list of files
                data.append(im)
                eval_files.append(files_p[k*pile:k*pile +pile]) # eval_files contains lists with all the files in a given pile 

        elif mode == "reverse" :
            """
                0-1 1-0 2-3 3-2 etc
            """
            # normal pile
            for k in range(len(files_p) // pile) : # number of subpile we can find
                im = np.zeros((1,im_ref.shape[0], im_ref.shape[1], pile))
                for idx,i in enumerate(range(k*pile, k*pile +pile)):
                    im[0,:,:,idx] = normalize_sar(np.load(files_p[i]))

                # add the pile images and the list of files
                data.append(im)
                eval_files.append(files_p[k*pile:k*pile +pile]) # eval_files contains lists with all the files in a given pile 
            # reverse pile
            for k in range(len(files_p) // pile) : # number of subpile we can find
                im = np.zeros((1,im_ref.shape[0], im_ref.shape[1], pile))
                for idx, i in enumerate(range(k*pile +pile-1 , k*pile -1,-1)):
                    im[0,:,:,idx] = normalize_sar(np.load(files_p[i]))

                # add the pile images and the list of files
                data.append(im)
                liste = files_p[k*pile:k*pile +pile].copy()
                liste.reverse()
                liste = [k.replace(os.path.basename(k), "reversed_"+ os.path.basename(k)) for k in liste]
                eval_files.append(liste) # eval_files contains lists with all the files in a given pile 

        elif mode == "all" :
            """
                0-1 0-2 0-3 0-4 0-5 0-6 etc
            """
            assert pile == 2, "Mode all not implemented for pile != 2 and you used pile = {}".format(pile)
            im = np.zeros((1,im_ref.shape[0], im_ref.shape[1], pile))
            im[0,:,:,0] = normalize_sar(np.load(files_p[0]))
            for idx in range(1,len(files_p)) :
                im[0,:,:,1] = normalize_sar(np.load(files_p[idx]))
                data.append(im.copy())
                eval_files.append([files_p[0].replace(os.path.basename(files_p[0]), "all_{}_".format(idx) + os.path.basename(files_p[0])), files_p[idx]])
        
        elif mode == "noise" :
            """
                0-noise 1-noise 2-noise etc
            """
            assert pile == 2, "Mode all not implemented for pile != 2 and you used pile = {}".format(pile)
            for k in range(len(files_p)) : # number of subpile we can find
                im = np.zeros((1,im_ref.shape[0], im_ref.shape[1], pile))
                im[0,:,:,0] = normalize_sar(np.load(files_p[k]))
                im[0,:,:,1] = np.random.normal(0., 1., size=im_ref.shape) # WGN or pure speckle?
                data.append(im)
                eval_files.append([files_p[k], files_p[k][:-4] + '_noise' + files_p[k][-4:]])

        else :
            assert 1==0, "Mode specified not found. You specified {} and only default, reverse, all and noise are available.".format(mode)
    return data, eval_files


"""def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = normalize_sar(np.load(filelist))
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = normalize_sar(np.load(file))
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1))
    return data"""





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

