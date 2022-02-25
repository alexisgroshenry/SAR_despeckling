import glob
import random
import os
import numpy as np

'''
Generate patches for the images in the folder dataset/data/Train
The code scans among the training images and then for data_aug_times
'''

class GenerateDataset():
    def data_augmentation(self, image, mode):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down
            return np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)


    def generate_patches(self,src_dir="./dataset/data/Train",pat_size=256,step=0,stride=128,bat_size=4,data_aug_times=1,n_channels=2):
        count = 0
        filepaths = glob.glob(src_dir + '/*.npy')
        print("number of training data %d" % len(filepaths))
        
        print(stride)
        print(pat_size)
        print(step)

        # calculate the number of patches
        for i in range(len(filepaths)):
            img = np.load(filepaths[i])
            print(img.shape)
            
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
       
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count * data_aug_times
        print(origin_patch_num)

        if origin_patch_num % bat_size != 0:
            numPatches = (origin_patch_num / bat_size + 1) * bat_size
        else:
            numPatches = origin_patch_num
        print("total patches = %d , batch size = %d, total batches = %d" % \
              (numPatches, bat_size, numPatches / bat_size))

        # data matrix 4-D
        numPatches=int(numPatches)
        inputs = np.zeros((numPatches, pat_size, pat_size, n_channels), dtype="float32")


        count = 0
        # generate patches
        for i in range(len(filepaths)): #scan through images
            img = np.load(filepaths[i])
            img_s = img


            # If data_aug_times = 8 then perform them all, otherwise pick one at random or do nothing
            for j in range(data_aug_times):
                im_h = np.size(img, 0)
                im_w = np.size(img, 1)
                if data_aug_times == 8:
                    for x in range(0 + step, im_h - pat_size, stride):
                        for y in range(0 + step, im_w - pat_size, stride):
                            inputs[count, :, :, :] = self.data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :], \
                                  j)
                            count += 1
                else:
                    for x in range(0 + step, im_h - pat_size, stride):
                        for y in range(0 + step, im_w - pat_size, stride):
                            # to pick one at random, uncomment this line and comment the one below
                            """inputs[count, :, :, :] = self.data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :], \
                                                                          random.randint(0, 7))"""


                            inputs[count, :, :, :] = self.data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :],0)

                            count += 1


        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

        return inputs

