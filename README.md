## Multi-temporal SAR images despeckling

This code proposes a temporal extension of the [SAR-CNN](https://arxiv.org/abs/2006.15559) method, to remove speckle noise in a temporal sequence of SAR images.

### Code

We describe here the main arguments to run the script `main.py` :
 - `--phase` takes value train or test
 - `--pile` corresponds to the size of the input sequence fed to the network
 - `--miso` if set to True, then the network will only output the first denoised image from the sequence. Otherwise, it will also output the change maps.
 - `--copy_input` if set to True, then the network is fed with the same image in input
 - `--load_all` if set to True, all training images are loaded on CPU beforehand (requires enough RAM)
 - `--test_config` choose between the following options when evaluatiing the model
    - `default` : images are loaded successively in the order of the folder
    - `reverse` : images are loaded in the same order as the previous option, but the sequence is inverted (only if `pile==2`)
    - `all` : the first image is fixed to the first image of each sequence, and the results are computed for all other images from the sequence in the second channel (only if `pile==2`)
    - `noise` : the first image is an image from the database, and the second image only contains pure white gaussian noise

The notebook `test.ipynb` is used to compute the metrics on the denoised images obtained when `phase==test`

### Data
The training code expects speckle free images on which artificial noise will be added, while the testing code expects images on which the noise is already present.