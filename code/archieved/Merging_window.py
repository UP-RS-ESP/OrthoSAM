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
    print('Loaded parameters from json')
    print(init_para)
except:#use defined init_para
    print('Using default parameters')
    print(init_para)

OutDIR=init_para.get('OutDIR')
DataDIR=init_para.get('DataDIR')
DSname=init_para.get('DatasetName')
fid=init_para.get('fid')

#defining clips
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')

image=fnc.load_image(DataDIR,DSname,fid)
print('Image size:', image.shape)

pre_para={'Downsample': {'fxy':resample_factor},
        #'Gaussian': {'kernel size':3}
        #'CLAHE':{'clip limit':2}#,
        #'Downsample': {'fxy':4},
        #'Buffering': {'crop size': crop_size}
        }
try:#attempt to load saved init_para
    with open(OutDIR+'pre_para.json', 'r') as json_file:
        pre_para = json.load(json_file)
    print('Loaded preprocessing parameters from json')
    print(pre_para)
except:#use defined init_para
    print('Using preprocessing default')
    print(pre_para)

image=fnc.preprocessing_roulette(image, pre_para)
print('Resampled to: ', image.shape)

print('Loading clips.....')

clips_pths = glob.glob(OutDIR+f'chunks/chunk_*')
clips_pths.sort()


print(len(clips_pths),' clips imported')


#nms_stops = [len(clips_pths) * i // 20 for i in range(1, 20)]
#nms_stops = np.arange(20,len(all_reseg),20).tolist()
shrink_t=2048
if np.max(image.shape[:2])>shrink_t:
    shrink_mask=np.max(image.shape[:2])//shrink_t
else:
    shrink_mask=False
msk_count=0

try:
    #Merging windows
    Aggregate_masks_noedge=[]
    pred_iou_noedge=[]
    for w_count,pth in tqdm(enumerate(clips_pths),f'Merging and resizing clips', total=len(clips_pths), unit='clips'):
        clip_window=np.load(pth, allow_pickle=True)[0]
        i=clip_window['i']
        j=clip_window['j']
        for mask,score in tqdm(zip(clip_window['nms mask'], clip_window['nms mask pred iou']), f'Merging and resizing masks in clip {i,j} (RAM: {fnc.get_memory_usage():.2f} MB, {msk_count} masks)',unit='masks',leave=False,total=len(clip_window['nms mask pred iou'])):
            if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                resize=np.zeros(image.shape[:-1])
                Valid_area=resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)].shape
                if Valid_area==(crop_size,crop_size):
                    resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask
                else:
                    resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
                if shrink_mask:
                    Aggregate_masks_noedge.append(fnc.resample_fnc(resize,{'fxy': 1/shrink_mask}).astype('bool'))
                else:
                    Aggregate_masks_noedge.append(resize.astype('bool'))
                pred_iou_noedge.append(score)
                msk_count+=1
        clip_window.clear()

        #if w_count in nms_stops:
        Aggregate_masks_noedge, pred_iou_noedge=fnc.nms(Aggregate_masks_noedge,pred_iou_noedge)
except Exception as error:
    print("An exception occurred:", error)

print(f'{msk_count} masks found')
Aggregate_masks_noedge_nms,_=fnc.nms(Aggregate_masks_noedge,pred_iou_noedge)
print(f'NMS filtered, {len(Aggregate_masks_noedge_nms)} masks remains')
del Aggregate_masks_noedge,pred_iou_noedge

stacked_Aggregate_masks_noedge_nms = np.zeros_like(Aggregate_masks_noedge_nms[0], dtype=np.uint8)
id_mask = np.zeros_like(Aggregate_masks_noedge_nms[0], dtype=np.uint16)
# Sum in chunks
for i,mask in enumerate(Aggregate_masks_noedge_nms):
    stacked_Aggregate_masks_noedge_nms += mask#.astype(np.uint8)
    id_mask[mask==1] = i
image=fnc.resample_fnc(image,{'fxy': 1/shrink_mask})
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.imshow(stacked_Aggregate_masks_noedge_nms,alpha=0.6)
plt.axis('off')
plt.title(f'No edge nms Stacked, max overlap: {np.max(stacked_Aggregate_masks_noedge_nms)}', fontsize=20)
plt.subplot(1,3,2)
plt.imshow(image)
plt.imshow(stacked_Aggregate_masks_noedge_nms!=0,alpha=0.6)
#plt.axis('off')
plt.title(f'No edge non zeros, {len(Aggregate_masks_noedge_nms)} mask(s)', fontsize=20)
plt.subplot(1,3,3)
plt.imshow(image)
plt.imshow(stacked_Aggregate_masks_noedge_nms>1,alpha=0.6)
plt.axis('off')
plt.title(f'Overlapping area after nms \nmasks downsampled by factor of {shrink_mask}', fontsize=20)
plt.tight_layout()
plt.savefig(OutDIR+'Merged_mask.png')
plt.show()

#calculate stats and save
print('Calculating stats')
stats=fnc.create_stats_df(Aggregate_masks_noedge_nms)
stats.to_hdf(OutDIR+'stats_df.h5', key='df', mode='w')
print('Stats saved')

from scipy.stats import gaussian_kde
loged = True 

plt.figure(figsize=(16, 10))
plt.subplot(2,2,1)
for df in [stats]:
    if loged:
        plt.xscale('log')
    data = df['area']

    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde(x)
    plt.plot(x, kde_values)

plt.xlabel('Area (pixel)')
plt.ylabel('Density')
plt.title('Density Plot of Area')
plt.grid()

plt.subplot(2,2,2)
loged = True
for df in [stats]:
    if loged:
        plt.xscale('log')
        #plt.yscale('log')
    data = df['area']

    frequencies, bin_edges = np.histogram(data, bins=30)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_midpoints, frequencies)

plt.xlabel('Area (pixel)')
plt.ylabel('Frequency')
plt.title('Frequency Plot of Area')
plt.grid()

loged=False
plt.subplot(2,2,3)
for df in [stats]:
    if loged:
        plt.xscale('log')
    data = df['area']

    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde(x)
    plt.plot(x, kde_values)

plt.xlabel('Area (pixel)')
plt.ylabel('Density')
plt.title('Density Plot of Area (nms)')
plt.grid()

plt.subplot(2,2,4)
loged = False
for df in [stats]:
    if loged:
        #plt.xscale('log')
        plt.yscale('log')
    data = df['area']

    frequencies, bin_edges = np.histogram(data, bins=30)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_midpoints, frequencies)

plt.xlabel('Area (pixel)')
plt.ylabel('Frequency')
plt.title('Frequency Plot of Area (nms)')
plt.grid()
plt.suptitle(f'{ len(stats)} object(s), masks downsampled by factor of {shrink_mask}')
plt.tight_layout()
plt.savefig(OutDIR+'size_distribution.png')
plt.show()

print(f'Saving id mask to '+OutDIR+'Merged/all_mask_merged_windows_id.npy...')
np.save(OutDIR+'Merged/all_mask_merged_windows_id',id_mask)
print('Saved')


if len(Aggregate_masks_noedge_nms)<1000:
    print(f'Saving id mask to '+OutDIR+'Merged/all_mask_merged_windows.npy...')
    saving_merged=[]
    for mask in Aggregate_masks_noedge_nms:
        saving_merged.append({'mask':mask.astype('bool')})
    np.save(OutDIR+'Merged/all_mask_merged_windows.npy',Aggregate_masks_noedge_nms)
else:
    batch_size=1000
    batches=len(Aggregate_masks_noedge_nms)//batch_size+1
    print(f'Splitting to {batches} saves')
    for i in range(batches):
        saving_merged=[]
        if i!=batches:
            for mask in Aggregate_masks_noedge_nms[i*batch_size:(i+1)*batch_size]:
                saving_merged.append({'mask':mask.astype('bool')})
        else:
            for mask in Aggregate_masks_noedge_nms[i*batch_size:]:
                saving_merged.append({'mask':mask.astype('bool')})
        np.save(OutDIR+f'Merged/all_mask_merged_windows_{i}.npy',saving_merged)
print('Saved')

end_script = time.time()
print('script took: ', end_script-start_script)
print('Merging windows completed. Output saved to '+OutDIR)