import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functions as fnc

min_radi=64
for std in [3,6,12]:

    Dir=f'/DATA/vito/data/ran_synth_{min_radi}_cl_std_{std:02}/'

    if not os.path.exists(Dir[:-1]):
        os.makedirs(Dir[:-1])
        os.makedirs(Dir+'img')
        os.makedirs(Dir+'msk')
    for i in range(10):

        image = np.load(f'/DATA/vito/data/ran_synth_{min_radi}_cl_std_00/img/img_{i:02}.npy')

        mask = np.load(f'/DATA/vito/data/ran_synth_{min_radi}_cl_std_00/msk/msk_{i:02}.npy')

        image=fnc.add_guassian_noise_to_circle(image, 0, std, mask=mask)
        image = np.clip(image, 0, 255).astype(np.uint8)

        np.save(Dir+f'img/img_{i:02}',image)
        np.save(Dir+f'msk/msk_{i:02}',mask)

        #plt.imshow(image)
        #plt.show(block=False)
        #plt.pause(3)
        #plt.close('all')