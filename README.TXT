# README

## Basic network :
- Train : python main.py
- Test : python main.py --phase test





To use an image stack, you have to give the size of the stack with the argument --pile

### SC_MAP
- Train : python main.py --pile 2
- Test : python main.py --phase test --pile 2

### SCM_SIAMESE
- Train : python main.py --pile 2
- Test : python main.py --phase test --pile 2





### UNET
- You need to adapt the output of the u_net in function of the method that you want to use, there is comments in u_net.py for that





### Images
- You have to provide noisy images in evaluation and test and not noisy images in training
- When you want to use network :
    -  With 2 different time t1 and t2 (or t2 and t1), you need to produce the corresponding noisy images 
    -  With the duplication of informations (the same image but with 2 different simulations of noise) you have to produce yourself the correspondant noisy images for the evaluation and/or test, the training part is automatic.
