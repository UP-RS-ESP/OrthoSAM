import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator_mod2 as SamAutomaticMaskGenerator
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
import json
import sys
import pandas as pd

start_script = time.time()

#load image
init_para={'OutDIR': '/DATA/vito/output/Ravi2_fnc_dw8/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'Ravi/*',
      'fid': 0,
      'crop_size': 1024,
      'resample_factor': 1/8,
      'dilation_size':15,
      'min_size_factor':0.0001
      }

try:#attempt to load saved init_para
    OutDIR=sys.argv[1]
    with open(OutDIR+'init_para.json', 'r') as json_file:
        init_para = json.load(json_file)
    print('Loaded parameters from json')
    print(init_para)
except:#use defined init_para
    print('Using default paramters')
    print(init_para)


DataDIR=init_para.get('DataDIR')
DSname=init_para.get('DatasetName')
fid=init_para.get('fid')

#defining clips
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')
dilation_size=init_para.get('dilation_size')
min_size_factor=init_para.get('min_size_factor')

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
print('Preprocessing finished')
#process related var
#original 5,5dilation_size=15
#for 3000x3000, 10000    ####need to be adaptive, consider 10% of image size
min_void_size=image.shape[0]*image.shape[1]*min_size_factor

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
#try:
#    all_reseg=np.load(OutDIR+'all_mask_merged_windows.npy', allow_pickle=True)
#except:
#    fn_save = glob.glob(OutDIR+'all_mask_merged_windows_*.npy')
#    fn_save.sort()
#    all_reseg=[]
#    for fn in fn_save[:-1]:
#        print(fn)
#        all_reseg+=np.load(fn, allow_pickle=True).tolist()
    
#ar_masks=all_reseg.astype(np.uint8)
ar_masks=np.array(np.load(OutDIR+'all_mask_merged_windows_id.npy', allow_pickle=True))
print(np.max(ar_masks),' mask(s) loaded')


#identify void
kernel = np.ones((dilation_size, dilation_size), np.uint8)
#stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(stacked_Aggregate_masks_noedge_nms>=1, kernel)
#no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)
stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(ar_masks>=1, kernel)
no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)

regions=regionprops(no_mask_area)

list_of_no_mask_area_centroid=[]
list_of_no_mask_area_mask=[]
list_of_no_mask_area_bbox=[]
for i,region in enumerate(regions):
    # take regions with large enough areas
    if (region.area > min_void_size):
        mask=np.array(no_mask_area==(i+1))
        #if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
        y0, x0 =region.centroid
        minr, minc, maxr, maxc = region.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        list_of_no_mask_area_centroid.append((y0,x0))
        list_of_no_mask_area_mask.append(no_mask_area==(i+1))
        list_of_no_mask_area_bbox.append([bx,by])

if len(list_of_no_mask_area_mask)>0:
    print(f'{len(list_of_no_mask_area_mask)} void(s) found')
    ar_no_mask_area=np.stack(list_of_no_mask_area_mask)

    plt.imshow(np.sum(ar_no_mask_area,axis=0), cmap='nipy_spectral')
    reseg_fxy=[]
    bxmax=0
    bxmin=9999
    bymax=0
    bymin=9999
    for i in range(ar_no_mask_area.shape[0]):
        y0, x0 =list_of_no_mask_area_centroid[i]
        bx, by =list_of_no_mask_area_bbox[i]
        if np.max(bx)>bxmax:
            bxmax=np.max(bx)
        if np.min(bx)<bxmin:
            bxmin=np.min(bx)
        if np.max(by)>bymax:
            bymax=np.max(by)
        if np.min(by)<bymin:
            bymin=np.min(by)

        plt.plot(x0, y0, '.g', markersize=15)
        plt.plot(bx, by, '-b', linewidth=2.5)
        width, length=np.max(bx)-np.min(bx),np.max(by)-np.min(by)
        print(f'Void {i} width: {width} length: {length}')
        if ((np.max([width,length]))<=(crop_size*0.8)):
            print(f'Downsample not necessary')
            reseg_fxy.append(1)
        else:
            factor=int((np.max([length,width]))//(crop_size*0.8))+1
            print(f'Downsampled by factor of {factor} required')
            reseg_fxy.append(factor)
    plt.title('Void in clean SAM masks')

    Big_width=bxmax-bxmin
    Big_length=bymax-bymin
    all_void_x0=bxmin+Big_width/2
    all_void_y0=bymin+Big_length/2
    plt.plot(all_void_x0, all_void_y0, '.r', markersize=15)
    plt.savefig(OutDIR+'void.png')
    plt.show()

    if ((np.max([Big_width,Big_length]))<=(crop_size*0.8)):
        print(f'Downsample not necessary for all voids to fit in one go')
        reseg_fxy.append(1)
    else:
        factor=int((np.max([Big_length,Big_width]))//(crop_size*0.8))+1
        print(f'Downsampled by factor of {factor} required for all voids to fit in one go')
        reseg_fxy.append(factor)

    if reseg_fxy[-1]<4:#if level of downsampling is small
        if reseg_fxy[-1]<=np.max(reseg_fxy[:-1]):
            print(f'Single downsample window can fit all voids, downsample by factor of {reseg_fxy[-1]}')
            onego=True
        else:
            onego=False
            print(f'Single downsample window cannot fit all voids or requires higher degree of downsampling')
    else:
        if reseg_fxy[-1]<=np.mean(reseg_fxy[:-1]):
            print(f'Single downsample window can fit all voids, downsample by factor of {reseg_fxy[-1]} required and on average requires less downsampling')
            onego=True
        else:
            print(f'Single downsample window can fit all voids but on average requires more downsampling')
            onego=False

    if onego==True:#do a comple reseg
        print('Performing third pass SAM in one go')
        fxy=1/reseg_fxy[-1]
        pre_para={'Downsample': {'fxy':fxy}}
        image_dw=fnc.preprocessing_roulette(image, pre_para)

        #define the window
        xmin,ymin=x0*fxy-crop_size/2,y0*fxy-crop_size/2
        ji=xmin/crop_size
        ii=ymin/crop_size
        if ji<0:
            ji=0
        if ii<0:
            ii=0

        #prepare image
        pre_para={'Downsample': {'fxy':fxy},
                'Crop': {'crop size': crop_size, 'j':ji,'i':ii},
                #'Gaussian': {'kernel size':3}
                # #'CLAHE':{'clip limit':2}#
                # #'Downsample': {'fxy':4},
                # #'Buffering': {'crop size': crop_size}
                    }
        temp_image=fnc.preprocessing_roulette(image, pre_para)

        #clear gpu ram and setup sam
        
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        predictor.set_image(temp_image)

        void_pointed_reseg=[]
        for i in range(ar_no_mask_area.shape[0]):
            y0, x0 =list_of_no_mask_area_centroid[i]
            input_point = np.array([[int(x0*fxy-xmin),int(y0*fxy-ymin)]])
            input_label = np.array([1])

            gc.collect()
            torch.cuda.empty_cache()

            partmasks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,)
            best_idx=np.argmax(scores)#pick the mask with highest score
            void_pointed_reseg.append(partmasks[best_idx])
        
        #merging reseg
        void_pointed_reseg_resized=[]
        for mask in void_pointed_reseg:
            if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                resize=np.zeros(image_dw.shape[:-1])
                Valid_area=resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)].shape
                if Valid_area==(crop_size,crop_size):
                    resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]=mask
                else:
                    resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
                #upsampling back to original resolution
                resize=fnc.resample_fnc(resize,{'target_size':image.shape[:-1][::-1]}).astype('bool')
                void_pointed_reseg_resized.append(resize)
        #if len(void_pointed_reseg_resized)!=0:
        #    ar_masks=np.vstack((ar_masks,np.stack(void_pointed_reseg_resized)))
        #list_of_mask_centroid = [fnc.get_centroid(ar_masks[i]) for i in range(ar_masks.shape[0])]
        del void_pointed_reseg
    else:
        print('Performing third pass SAM per void')
        #void_pointed_reseg=[]
        void_pointed_reseg_resized=[]
        for i in range(ar_no_mask_area.shape[0]):
            y0, x0 =list_of_no_mask_area_centroid[i]
            fxy=1/reseg_fxy[i]
            pre_para={'Downsample': {'fxy':fxy}}
            image_dw=fnc.preprocessing_roulette(image, pre_para)
            xmin,ymin=x0*fxy-crop_size/2,y0*fxy-crop_size/2
            ji=xmin/crop_size
            ii=ymin/crop_size
            if ji<0:
                ji=0
            if ii<0:
                ii=0

            #prepare image
            pre_para={'Downsample': {'fxy':fxy},
                    'Crop': {'crop size': crop_size, 'j':ji,'i':ii},
                    #'Gaussian': {'kernel size':3}
                    # #'CLAHE':{'clip limit':2}#
                    # #'Downsample': {'fxy':4},
                    # #'Buffering': {'crop size': crop_size}
                        }
            temp_image=fnc.preprocessing_roulette(image, pre_para)

            #clear gpu ram and setup sam
            gc.collect()
            torch.cuda.empty_cache()
            sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
            sam.to(device=DEVICE)
            predictor = SamPredictor(sam)
            predictor.set_image(temp_image)

            input_point = np.array([[int(x0*fxy-xmin),int(y0*fxy-ymin)]])
            input_label = np.array([1])

            partmasks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,)
            best_idx=np.argmax(scores)#pick the mask with highest score
            #void_pointed_reseg.append(partmasks[best_idx])
            mask=partmasks[best_idx]
            
            #for mask in void_pointed_reseg:
            if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                resize=np.zeros(image_dw.shape[:-1])
                Valid_area=resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)].shape
                if Valid_area==(crop_size,crop_size):
                    resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]=mask
                else:
                    resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
                resize=fnc.resample_fnc(resize,{'target_size':image.shape[:-1][::-1]}).astype('bool')
                void_pointed_reseg_resized.append(resize)
            #if len(void_pointed_reseg_resized)!=0:
            #    ar_masks=np.vstack((ar_masks,np.stack(void_pointed_reseg_resized)))
            #list_of_mask_centroid = [fnc.get_centroid(ar_masks[i]) for i in range(ar_masks.shape[0])]


    if len(void_pointed_reseg_resized)!=0:
        id_mask = np.sum([void_pointed_reseg_resized[i]*(i+1) for i in range(len(void_pointed_reseg_resized))],axis=0)
        print(f'Saving id mask to '+OutDIR+'all_mask_merged_windows_id.npy...')
        np.save(OutDIR+'all_mask_thid_pass_id',id_mask)
        print('Saved')
        #saving mask
        if len(void_pointed_reseg_resized)<1000:
            print(f'Saving id mask to '+OutDIR+'all_mask_merged_windows_void_filled.npy...')
            saving_merged=[]
            for mask in range(len(void_pointed_reseg_resized)):
                saving_merged.append({'mask':void_pointed_reseg_resized[mask].astype('bool')})
            np.save(OutDIR+'all_mask_third_pass.npy',saving_merged)
        else:
            batch_size=1000
            batches=len(void_pointed_reseg_resized)//batch_size+1
            print(f'Splitting to {batches} saves')
            for i in range(batches):
                saving_merged=[]
                if i!=batches:
                    for msk in np.arange(i*1000,(i+1)*1000):
                        saving_merged.append({'mask':void_pointed_reseg_resized[i].astype('bool')})
                else:
                    for msk in np.arange(i*1000,len(void_pointed_reseg_resized)):
                        saving_merged.append({'mask':void_pointed_reseg_resized[i].astype('bool')})
                np.save(OutDIR+f'all_mask_third_pass_{i}.npy',saving_merged)
        print('Saved')

        plt.figure(figsize=(20,20))
        plt.subplot(2,2,1)
        plt.imshow(image)
        plt.imshow((np.sum(void_pointed_reseg_resized,axis=0)),alpha=0.6)
        plt.axis('off')
        plt.title(f'No edge nms Stacked after\nmerging downsampled mask\nmax overlap: {np.max(np.sum(void_pointed_reseg_resized,axis=0))}', fontsize=20)
        plt.subplot(2,2,2)
        plt.imshow(image)
        plt.imshow((ar_masks+np.sum(void_pointed_reseg_resized,axis=0))>0,alpha=0.6)
        plt.axis('off')
        plt.title('Masked area after nms and\nmerging downsampled mask', fontsize=20)
        plt.subplot(2,2,3)
        plt.imshow(ar_masks+id_mask, cmap='nipy_spectral')
        plt.axis('off')
        plt.title(f'Mask area after nms and\nmerging downsampled mask\n No. of mask: {np.max(ar_masks)+len(void_pointed_reseg_resized)}', fontsize=20)
        plt.subplot(2,2,4)
        plt.imshow(image)
        plt.imshow((np.sum(void_pointed_reseg_resized,axis=0)+ar_masks!=0)>1,alpha=0.6)
        plt.axis('off')
        plt.title('Overlapping area after nms and\nmerging downsampled mask', fontsize=20)
        plt.tight_layout()
        plt.savefig(OutDIR+'Merged_masks_withvoid.png')
        plt.show()

        #calculate stats and save
        print('Calculating stats')
        stats=fnc.create_stats_df(void_pointed_reseg_resized)
        loaded_stats = pd.read_hdf(OutDIR + 'stats_df.h5')
        max_label = loaded_stats['label'].max()
        stats['label'] += max_label
        stats = pd.concat([stats, loaded_stats], ignore_index=True)
        stats.to_hdf(OutDIR+'stats_df_thirdpass.h5', key='df', mode='w')
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
        plt.legend()
        plt.grid()
        plt.suptitle(f'Thid_pass { len(np.unique(stats))} object(s)')
        plt.tight_layout()
        plt.savefig(OutDIR+'size_distribution_thidpass.png')
        plt.show()
    else:
        print('Void(s) identified but no valid mask was found')
else:
    print('No void found. Minimum size threshold or dilation parameter may need to be adjusted')
end_script = time.time()
print('script took: ', end_script-start_script)
print('Third pass SAM completed. Output saved to '+OutDIR)
