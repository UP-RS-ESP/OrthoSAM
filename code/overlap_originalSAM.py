import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import functions as fnc
from importlib import reload
import gc
from skimage.measure import label, regionprops
from collections import Counter
from torchvision.ops.boxes import batched_nms
from sklearn.neighbors import KDTree
import matplotlib.colors as mcolors
import os
import time
from skimage.morphology import binary_dilation

start_script = time.time()
#load image
OutDIR='/DATA/vito/output/original_SAM/'
if not os.path.exists(OutDIR[:-1]):
    os.makedirs(OutDIR[:-1])
DataDIR='/DATA/vito/data/'

image=(np.load(DataDIR+'example/rgb.npy')*255).astype(np.uint8)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
seg_ids=np.load(DataDIR+'example/segment_ids.npy')

#setup SAM
MODEL_TYPE = "vit_h"
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print('Currently running on GPU\nModel '+MODEL_TYPE)
else:
    DEVICE = torch.device('cpu')
    print('Currently running on CPU\nModel '+MODEL_TYPE)

if MODEL_TYPE == 'vit_h':
    CHECKPOINT_PATH = DataDIR+'MetaSAM/sam_vit_h_4b8939.pth'
elif MODEL_TYPE == 'vit_l':
    CHECKPOINT_PATH = DataDIR+'MetaSAM/sam_vit_l_0b3195.pth'
else:
    CHECKPOINT_PATH = DataDIR+'MetaSAM/sam_vit_b_01ec64.pth'

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

#defining clips
crop_size=1024
downsample_factor=1
clipi=np.arange(0,(image.shape[0]*downsample_factor)//crop_size+1,0.5)
clipj=np.arange(0,(image.shape[1]*downsample_factor)//crop_size+1,0.5)
clipij=np.array(np.meshgrid(clipi, clipj)).T.reshape(-1,2)


#containers
all_reseg=[]

for ij_idx in clipij:
    start_loop = time.time()
    print(f'Clip: {ij_idx}')
    ji=ij_idx[1]
    ii=ij_idx[0]
    if (ji*crop_size<(image.shape[1]*downsample_factor) and ii*crop_size<(image.shape[0]*downsample_factor)):
        #prepare image
        pre_para={'Downsample': {'fxy':downsample_factor},
                'Crop': {'crop size': crop_size, 'j':ji,'i':ii},
                'Gaussian': {'kernel size':3}
                #'CLAHE':{'clip limit':2}#,
                #'Downsample': {'fxy':4},
                #'Buffering': {'crop size': crop_size}
                }
        temp_image=fnc.preprocessing_roulette(image, pre_para)
        if (temp_image.shape[0]>500 and temp_image.shape[1]>500):

            #clear gpu ram
            gc.collect()
            torch.cuda.empty_cache()

            #SAM segmentation
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=24,
                pred_iou_thresh=0.95,
                stability_score_thresh=0.95,#iou by varying cutoff in binary conversion
                box_nms_thresh=0.3,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks
                crop_n_layers=0,#cut into 2**n crops
                crop_nms_thresh=0,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks between crops
                crop_n_points_downscale_factor=1,
                crop_overlap_ratio=0,
                #min_mask_region_area=2000,
            )
            predictor = SamPredictor(sam)
            predictor.set_image(temp_image)

            with torch.no_grad():
                masks = mask_generator.generate(temp_image)
            print(len(masks))

            #saving outputs
            all_reseg.append({'mask':masks,
                            'i':ii,
                            'j':ji,
                            'crop size':crop_size})
            np.save(OutDIR+'all_reseg_mask',np.hstack(all_reseg))
            
        else:
            print('This crop is too small')
    else:
        print('Exceeded image boundary')
    end_loop = time.time()
    print('loop took: ', end_loop-start_loop)
    
#release ram
del masks

#Merging windows
ALL_seg=[]
crop_size=1024
for clip_window in all_reseg:
    i=clip_window['i']
    j=clip_window['j']
    for mask,score in zip([mask['segmentation'] for mask in clip_window['mask']], [mask['predicted_iou'] for mask in clip_window['mask']]):
        if (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
            edge=True
        else:
            edge=False
        resize=np.zeros(image.shape[:-1])
        Valid_area=resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)].shape
        if Valid_area==(crop_size,crop_size):
            resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask
        else:
            resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
        ALL_seg.append({'mask':resize,'pred_iou':score,'edge':edge})
np.save(OutDIR+'all_resized',np.hstack(ALL_seg))

end_script = time.time()
print('script took: ', end_script-start_script)