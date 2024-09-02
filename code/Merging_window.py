import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import functions as fnc
import time
import sys
import json

start_script = time.time()

#process related var

#load image
init_para={'OutDIR': '/DATA/vito/output/Ravi2_fnc_dw8/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'Ravi/*',
      'fid': 0,
      'crop_size': 1024,
      'resample_factor': 1/8
      }

try:#attempt to load saved init_para
    OutDIR=sys.argv[1]
    with open(OutDIR+'init_para.json', 'r') as json_file:
        init_para = json.load(json_file)
    print('Loaded json')
    print(init_para)
except:#use defined init_para
    print('Using default')
    print(init_para)

OutDIR=init_para.get('OutDIR')
DataDIR=init_para.get('DataDIR')
fn_img = glob.glob(DataDIR+init_para.get('DatasetName'))
fid=init_para.get('fid')

#defining clips
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')

fn_img.sort()
image = cv2.imread(fn_img[fid])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image#[100:-200,500:-1000]
print(fn_img[fid].split("/")[-1]+' imported')
print('Image size:', image.shape)

pre_para={'Downsample': {'fxy':resample_factor},
        #'Gaussian': {'kernel size':3}
        #'CLAHE':{'clip limit':2}#,
        #'Downsample': {'fxy':4},
        #'Buffering': {'crop size': crop_size}
        }
try:#attempt to load saved init_para
    with open(OutDIR+'pre_para.json', 'r') as json_file:
        init_para = json.load(json_file)
    print('Loaded preprocessing parameters from json')
    print(pre_para)
except:#use defined init_para
    print('Using preprocessing default')
    print(pre_para)

image=fnc.preprocessing_roulette(image, pre_para)

all_reseg=np.load(OutDIR+'all_reseg_mask.npy', allow_pickle=True)

#Merging windows
Aggregate_masks_noedge=[]
pred_iou_noedge=[]
for clip_window in all_reseg:
    i=clip_window['i']
    j=clip_window['j']
    for mask,score in zip(clip_window['nms mask'], clip_window['nms mask pred iou']):
        if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
            resize=np.zeros(image.shape[:-1])
            Valid_area=resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)].shape
            if Valid_area==(crop_size,crop_size):
                resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask
            else:
                resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
            Aggregate_masks_noedge.append(resize.astype(np.uint8))
            pred_iou_noedge.append(score)

print(f'{len(Aggregate_masks_noedge)} masks found')
Aggregate_masks_noedge_nms,_=fnc.nms(Aggregate_masks_noedge,pred_iou_noedge)
print(f'NMS filtered, {len(Aggregate_masks_noedge_nms)} masks remains')

stacked_Aggregate_masks_noedge_nms = np.zeros_like(Aggregate_masks_noedge_nms[0], dtype=np.uint8)

# Sum in chunks
for mask in Aggregate_masks_noedge_nms:
    stacked_Aggregate_masks_noedge_nms += mask.astype(np.uint8)

plt.figure(figsize=(20,15))
plt.subplot(1,3,1)
plt.imshow(image)
plt.imshow(stacked_Aggregate_masks_noedge_nms,alpha=0.6)
plt.axis('off')
plt.title(f'No edge nms Stacked, max overlap: {np.max(stacked_Aggregate_masks_noedge_nms)}', fontsize=20)
plt.subplot(1,3,2)
plt.imshow(image)
plt.imshow(stacked_Aggregate_masks_noedge_nms!=0,alpha=0.6)
#plt.axis('off')
plt.title('No edge non zeros', fontsize=20)
plt.subplot(1,3,3)
plt.imshow(image)
plt.imshow(stacked_Aggregate_masks_noedge_nms>1,alpha=0.6)
plt.axis('off')
plt.title('Overlapping area after nms', fontsize=20)
plt.tight_layout()
plt.savefig(OutDIR+'Merged_mask.png')
plt.show()

print('Saving masks.....')
ar_masks=np.stack(Aggregate_masks_noedge_nms).astype(np.uint8)
saving_merged=[]
for mask in ar_masks:
    saving_merged.append({'mask':mask})
np.save(OutDIR+'all_mask_merged_windows',np.hstack(saving_merged))
print('Saved')

end_script = time.time()
print('script took: ', end_script-start_script)