import numpy as np
import os


min_radi=8
Dir=f'/DATA/vito/data/ran_synth_{min_radi:02}_bw_rt/'

if not os.path.exists(Dir[:-1]):
    os.makedirs(Dir[:-1])
    os.makedirs(Dir+'img')
    os.makedirs(Dir+'msk')
for i in range(10):
    image = np.load(f'/DATA/vito/data/ran_synth_{min_radi:02}_bw/img/img_{i:02}.npy')
    image=np.rot90(image,2)
    np.save(Dir+f'img/img_{i:02}',image)
    
    mask = np.load(f'/DATA/vito/data/ran_synth_{min_radi:02}_bw/msk/msk_{i:02}.npy')
    mask=np.rot90(mask,2)
    np.save(Dir+f'msk/msk_{i:02}',mask)

    #plt.imshow(image)
    #plt.show(block=False)
    #plt.pause(3)
    #plt.close('all')