import time
import numpy as np
import os
import random

from utils import *
from u_net import *
from scipy import special

M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)

class denoiser(object):
    def __init__(self, sess, stride=128, input_c_dim=1, miso=True, copy_input=False, batch_size=4, load_all=True):

        self.sess = sess
        self.input_c_dim = input_c_dim
        self.miso = miso
        self.copy_input = copy_input
        self.load_all = load_all
        self.X_input = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='real_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')



        def true_fn():
           s = tf.zeros(shape=tf.shape(self.X_input))
           for k in range(0, L):
               gamma = (tf.abs(tf.complex(tf.random_normal(shape=tf.shape(self.X_input), stddev=1),
                                          tf.random_normal(shape=tf.shape(self.X_input), stddev=1))) ** 2) / 2
               s = s + gamma
           s_amplitude = tf.sqrt(s / L)
           log_speckle = tf.log(s_amplitude)
           log_norm_speckle = log_speckle / (M - m)

           return tf.add(self.X_input, log_norm_speckle)

        def false_fn():
           return self.X_input


        self.X = tf.cond(self.is_training, true_fn, false_fn)
        self.Y  = autoencoder(self.X, self.input_c_dim, miso=self.miso)

        # ----- loss -----
        self.alpha_noise = 1.
        self.alpha_chge_map = 1.
        # self.loss = self.alpha*((1.0 / batch_size) * tf.reduce_sum( tf.abs( self.X_input[:,:,:,:1] - self.Y)))
        # sum of L1 losses between all images and the denoised image
        self.loss = self.alpha_noise*(1.0 / batch_size) * tf.reduce_sum( tf.abs(self.X_input[:,:,:,:1] - self.Y[:,:,:,:1]))

        if self.Y.shape[3] > 1 :
            self.loss += self.alpha_chge_map*(1.0 / batch_size) * tf.reduce_sum(tf.abs(self.X_input[:,:,:,1:] - self.Y[:,:,:,:1] - self.Y[:,:,:,1:]))

####################################################################    
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.grad_check = [tf.norm(gradient) for gradient in gradients]
            for gradient in gradients: self.print_op = tf.print(tf.norm(gradient),output_stream=sys.stderr)
            gradients = [
                None if gradient is None else tf.clip_by_norm(gradient,5.0)
                for gradient in gradients]
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
####################################################################   


    def evaluate(self, iter_num, test_data, eval_files, eval_set, sample_dir): 
        print("[*] Evaluating...")

        for idx, current_test in enumerate(test_data):
            real_image = current_test.astype(np.float32)

            real_image_channels=real_image[:,:256,:256,:]

            output_clean_image, noisy_image = self.sess.run(
                [self.Y, self.X],
                feed_dict={self.X_input: real_image_channels,
                           self.is_training: False})

            # modify evaluate to handle new format of eval_files (list of lists instead of list)
            for idx_p, file in enumerate(eval_files[idx]):
                if idx_p and self.miso:
                    # we stop here since the output is 1-dim
                    break
                groundtruth = denormalize_sar(real_image[:, :256, :256, idx_p])
                noisyimage = denormalize_sar(noisy_image[:,:,:,idx_p])
                if idx_p:
                    outputimage = denormalize_sar(output_clean_image[:,:,:,0]) + denormalize_sar(output_clean_image[:,:,:,idx_p])
                else:
                    outputimage = denormalize_sar(output_clean_image[:,:,:,0])
                imagename0 = file.replace(eval_set, "")
                imagename = imagename0.replace('.npy', '_' + str(iter_num) + '.npy')
                save_sar_images(outputimage, noisyimage, imagename, sample_dir)
        print("--- Evaluation ---- Done ---")
        
        
        
    def train(self, data, eval_data, eval_files, eval_set, batch_size, ckpt_dir, epoch, lr, sample_dir, step, pat_size, stride, eval_every_epoch=2):
        
####################################################################     
        numPatch_ = 0
        if not self.load_all :
            # If the data is to be loaded at each batch
            for i in range(data.shape[0]):
                count = 0
                im_h = data[i][2] 
                im_w = data[i][3] 

                for x in range(0 + step, im_h - pat_size, stride):
                    for y in range(0 + step, im_w - pat_size, stride):
                        count += 1
                numPatch_ += count* len(data[i][1])

            numPatch = int((numPatch_)/batch_size) * batch_size
            numBatch = int(numPatch / batch_size)
            

            indexes = np.zeros((numPatch, 4+1+1), dtype=np.uint16) 
            count_ = np.zeros(data.shape[0], dtype=np.uint16)
            for i in range(data.shape[0]): 
                im_h = data[i][2] 
                im_w = data[i][3]                
            
                for id_pile in range(len(data[i][1])) : 
                    count_b = 0
                    for x in range(0 + step, im_h - pat_size, stride):
                        for y in range(0 + step, im_w - pat_size, stride):
                            if np.sum(count_)<numPatch:
                                indexes[np.sum(count_),:] = [np.sum(count_),i,count_b, x, y, id_pile]
                                count_[i] += 1
                                count_b += 1

        if self.load_all :
            # data already in data
            for i in range(data.shape[0]):
                count = 0
                img = data[i][1][:,:,0] 
                im_h = np.size(img, 0) 
                im_w = np.size(img, 1)

                for x in range(0 + step, im_h - pat_size, stride):
                    for y in range(0 + step, im_w - pat_size, stride):
                        count += 1
                numPatch_ += count* data[i][1].shape[-1] 

            numPatch = int((numPatch_)/batch_size) * batch_size
            numBatch = int(numPatch / batch_size)
            

            indexes = np.zeros((numPatch, 4+1+1), dtype=np.uint16) 
            count_ = np.zeros(data.shape[0], dtype=np.uint16)
            for i in range(data.shape[0]): 
                im_h = np.size(data[i][1][:,:,0], 0)
                im_w = np.size(data[i][1][:,:,0], 1)
            
                for id_pile in range(data[i][1].shape[-1]): 
                    count_b = 0
                    for x in range(0 + step, im_h - pat_size, stride):
                        for y in range(0 + step, im_w - pat_size, stride):
                            if np.sum(count_)<numPatch:
                                indexes[np.sum(count_),:] = [np.sum(count_),i,count_b, x, y, id_pile]
                                count_[i] += 1
                                count_b += 1

                                
        np.random.shuffle(indexes)
        indexes[:,0] = np.arange(0, indexes.shape[0])

        
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // (numBatch)
            start_step = global_step % (numBatch)
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data, eval_files, eval_set=eval_set, sample_dir=sample_dir)

####################################################################   
        
        
        
        
        batch_images = np.zeros((batch_size, pat_size, pat_size, self.input_c_dim))
        batch_maps = np.zeros((batch_size, pat_size, pat_size, 1))
        print(numBatch)
        for epoch in range(start_epoch, epoch):
            
            for batch_id in range(start_step, numBatch):
                
                
                for i in range(batch_size):
                    id_pile = indexes[batch_id * batch_size + i, 1]
                    id_patch = indexes[batch_id * batch_size + i, 2]
                    x = indexes[batch_id * batch_size + i, 3]
                    y = indexes[batch_id * batch_size + i, 4]
                    id_date = indexes[batch_id * batch_size + i, -1]

                    stride_shift_x = np.random.randint(low=0, high=int(stride/2))
                    x = np.max((0,x-stride_shift_x))

                    stride_shift_y = np.random.randint(low=0, high=int(stride/2))
                    y = np.max((0,y-stride_shift_y))

                    """# Il prend une image à une date donnée
                    im0 = data[id_pile][1][ x:x + pat_size, y:y + pat_size, id_date]
                    im_channels=im0
                    # Il la duplique sur les channels de pile mais on veut pas ça, on aimerait des images différentes à chaque fois !!
                    im_channels = np.expand_dims(im_channels, axis=2) 
                    batch_images[i,:,:,:] = im_channels"""
                    if not self.load_all :
                        print('what')
                        if self.copy_input:
                            im0 = np.load(data[id_pile][1][id_date])[x:x+pat_size,y:y+pat_size]
                            im_channels=im0
                            # put twice the image in the batch
                            im_channels = np.expand_dims(im_channels, axis=2)
                            batch_images[i,:,:,:] = im_channels
                        else:
                            # a different image at each ~ date 
                            indices_chosen = [id_date]
                            for date in range(self.input_c_dim) : # for each stack
                                # if we do not want consecutives dates
                                if date == 0 :
                                    im0 = np.load(data[id_pile][1][id_date])[x:x+pat_size,y:y+pat_size]
                                else :
                                    new_idx = id_date
                                    while new_idx in indices_chosen :
                                        new_idx = random.randint(0, len(data[id_pile][1])-1)
                                    im0 = np.load(data[id_pile][1][new_idx])[x:x+pat_size,y:y+pat_size]
                                    indices_chosen.append(new_idx)
                                #if we want consecuitives dates
                                # im0 = np.load(data[id_pile][1][(id_date+date) % len(data[id_pile][1])])[x:x+pat_size,y:y+pat_size]
                                batch_images[i,:,:,date] = im0 # a date is a single image

                    if self.load_all :
                        if self.copy_input:
                            im0 = data[id_pile][1][ x:x + pat_size, y:y + pat_size, (id_date) % data[id_pile][1].shape[-1]] 
                            im_channels=im0
                            # put twice the image in the batch
                            im_channels = np.expand_dims(im_channels, axis=2)
                            batch_images[i,:,:,:] = im_channels
                        else:
                            # a different image at each ~ date 
                            indices_chosen = [id_date]
                            for date in range(self.input_c_dim) : # pour chaque pile 
                                # if we do not want consecutives dates
                                '''if date == 0 : 
                                    im0 = data[id_pile][1][ x:x + pat_size, y:y + pat_size, (id_date) % data[id_pile][1].shape[-1]] 
                                else :
                                    new_idx = id_date 
                                    while new_idx in indices_chosen : 
                                        new_idx = random.randint(0, len(data[id_pile][1])-1)
                                    im0 = data[id_pile][1][ x:x + pat_size, y:y + pat_size, (new_idx) % data[id_pile][1].shape[-1]] 
                                    indices_chosen.append(new_idx)'''
                                #if we want consecuitives dates
                                im0 = data[id_pile][1][ x:x + pat_size, y:y + pat_size, (id_date + date) % data[id_pile][1].shape[-1]] # image at index id_data + 0 etc 
                                batch_images[i,:,:,date] = im0 # a date is a single image

                _, loss,_= self.sess.run([self.train_op, self.loss, self.print_op],
                                         feed_dict={self.X_input: batch_images, self.lr: lr[epoch], self.is_training: True})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, (batch_id + 1) , numBatch, time.time() - start_time, loss)) 

                iter_num += 1 
    
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data, eval_files, eval_set=eval_set, sample_dir=sample_dir)
                self.save(iter_num, ckpt_dir)

        print("[*] Finish training.")
        
        
        
        
        
        
        
        
        
####################################################################      

    def save(self, iter_num, ckpt_dir, model_name='Unet_L1_PRAT'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)
        

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

####################################################################   















    def test(self, test_set, ckpt_dir, save_dir, pile, mode="default"): 
        tf.initialize_all_variables().run()
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        print("[*] start testing...")

        test_data, test_files = load_sar_images(test_set, pile, mode=mode)
        for idx in range(len(test_files)):
            real_image = test_data[idx].astype(np.float32)
            # real_image = real_image[:,:256,:256,:]

            stride = 32
            pat_size = 256

            im_h = np.size(real_image, 1)
            im_w = np.size(real_image, 2)

            count_image = np.zeros(real_image.shape)
            output_clean_image = np.zeros(real_image.shape)

            if im_h == pat_size:
                x_range = list(np.array([0]))
            else:
                x_range = list(range(0, im_h - pat_size, stride))
                if (x_range[-1] + pat_size) < im_h: x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

            if im_w == pat_size:
                y_range = list(np.array([0]))
            else:
                y_range = list(range(0, im_w - pat_size, stride))
                if (y_range[-1] + pat_size) < im_w: y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))

            for x in x_range:
                for y in y_range:
                    tmp_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                                 feed_dict={self.X_input: real_image[:, x:x + pat_size,
                                                                                     y:y + pat_size, :],
                                                                            self.is_training: False})
                    output_clean_image[:, x:x + pat_size, y:y + pat_size, :] += tmp_clean_image
                    # complete the change maps with the reference denoised image
                    if not self.miso:
                        output_clean_image[:, x:x + pat_size, y:y + pat_size, 1:] += tmp_clean_image[:,:,:,0][:,:,:,np.newaxis]
                    count_image[:, x:x + pat_size, y:y + pat_size, :] += np.ones((1, pat_size, pat_size, 1))
            output_clean_image = output_clean_image / count_image
            noisyimage = denormalize_sar(real_image)
            outputimage = denormalize_sar(output_clean_image)

            # handle the new format of test_files
            for idx_p, file in enumerate(test_files[idx]):
                if idx_p and self.miso:
                    break
                imagename = file.replace(test_set, "")
                save_sar_images(outputimage[:,:,idx_p], noisyimage[:,:,idx_p], imagename, save_dir)


        print("--- Done Testing ---")
