import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import functions as fnc
import time
import sys
import json
from tqdm import tqdm
import pickle
import psutil

start_script = time.time()

try:
    OutDIR=sys.argv[1]
except:
    OutDIR='/DATA/vito/output/Ravi2_run2_dw8_cp1024_pps48/'

print('Loaded parameters from '+OutDIR)
with open(OutDIR+'init_para.json', 'r') as json_file:
    init_para = json.load(json_file)
with open(OutDIR+'pre_para.json', 'r') as json_file:
    pre_para = json.load(json_file)

print(init_para)
print(pre_para)


OutDIR=init_para.get('OutDIR')
DataDIR=init_para.get('DataDIR')
DSname=init_para.get('DatasetName')
fid=init_para.get('fid')

#defining clips
b=init_para.get('b')
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')

image=fnc.load_image(DataDIR,DSname,fid)
org_shape=image.shape
print('Image size:', image.shape)

image=fnc.preprocessing_roulette(image, pre_para)
print('Resampled to: ', image.shape)

print('Loading clips.....')

clips_pths = glob.glob(OutDIR+f'chunks/chunk_*')
clips_pths.sort()


print(len(clips_pths),' clips imported')

msk_count=0
id_mask = np.zeros_like(image[:,:,0], dtype=np.uint16)
stack_mask = np.zeros_like(image[:,:,0], dtype=np.uint16)
try:
    #Merging windows
    Aggregate_masks_noedge=[]
    pred_iou_noedge=[]
    for w_count,pth in tqdm(enumerate(clips_pths),f'Merging and resizing clips', total=len(clips_pths), unit='clips'):
        clip_window=np.load(pth, allow_pickle=True)[0]
        i,j=clip_window['ij']

        for mask,score in tqdm(zip(clip_window['nms mask'], clip_window['nms mask pred iou']), f'Merging and resizing masks in clip {i,j} (RAM: {fnc.get_memory_usage():.2f} MB, {msk_count} masks)',unit='masks',leave=False,total=len(clip_window['nms mask pred iou'])):
            if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                resized = fnc.untile(id_mask, mask, i, j, crop_size, 2*b)
                msk_count+=1
                id_mask[resized!=0]=(msk_count)
                stack_mask+=resized
                #pred_iou_noedge.append(score)
        clip_window.clear()
except Exception as error:
    print("An exception occurred:", error)

#shuffle id
unique_labels = np.unique(id_mask)
if 0 in unique_labels:
    unique_labels = unique_labels[unique_labels != 0]
shuffled_labels = np.random.permutation(unique_labels)
label_mapping = dict(zip(unique_labels, shuffled_labels))
shuffled_mask = id_mask.copy()
for old_label, new_label in label_mapping.items():
    shuffled_mask[id_mask == old_label] = new_label
id_mask=shuffled_mask

print(f'Saving id mask to '+OutDIR+'Merged/all_mask_merged_windows_id.npy...')
np.save(OutDIR+'Merged/all_mask_merged_windows_id',id_mask)
print('Saved')

print(f'{msk_count} masks found')

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.imshow(stack_mask,alpha=0.6)
plt.axis('off')
plt.title(f'Max overlap: {np.max(stack_mask)}', fontsize=20)
plt.subplot(1,3,2)
plt.imshow(image)
plt.imshow(id_mask, cmap='nipy_spectral',alpha=0.5)
plt.title(f'Segments, {msk_count} mask(s)', fontsize=20)
plt.subplot(1,3,3)
plt.imshow(image)
plt.imshow(stack_mask>1,alpha=0.6)
plt.axis('off')
plt.title(f'Overlapping area after nms', fontsize=20)
plt.tight_layout()
plt.savefig(OutDIR+'Merged_mask.png')
plt.show()

end_script = time.time()
print('script took: ', end_script-start_script)
print('Merging windows completed. Output saved to '+OutDIR)