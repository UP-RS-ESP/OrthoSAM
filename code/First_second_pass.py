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
import sys
import json
start_script = time.time()

#vars
init_para={'OutDIR': '/DATA/vito/output/Ravi2_fnc_dw8/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'Ravi/*',
      'fid': 0,
      'crop_size': 1024,
      'resample_factor': 1/8,
      'point_per_side': 24,
      'window_step':0.5
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
    # create dir if output dir not exist
    if not os.path.exists(OutDIR[:-1]):
        os.makedirs(OutDIR[:-1])

OutDIR=init_para.get('OutDIR')
DataDIR=init_para.get('DataDIR')
DSname=init_para.get('DatasetName')
fid=init_para.get('fid')
pps=init_para.get('point_per_side')
window_step=init_para.get('window_step')

#defining clips
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')

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

#Prep
#load image
image=fnc.load_image(DataDIR,DSname,fid)
print('Image size:', image.shape)
clipij=fnc.define_clips(image.shape[0],image.shape[1],resample_factor,crop_size,window_step)
image=fnc.preprocessing_roulette(image, pre_para)

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

def filter_by_pred_iou_and_size_per_seedpoint(masks,size_threshold=0.4,crop_size=crop_size):
    seed_point=np.array([mask['point_coords'][0] for mask in masks])
    highest_pred_iou_by_point=[]
    i=0
    while i < len(masks):
        i0=i
        if np.all(seed_point[i]==seed_point[i+2]):
            i+=2
        elif np.all(seed_point[i]==seed_point[i+1]):
            i+=1
        iou=[mask['predicted_iou'] for mask in masks[i0:i+1]]
        idx=np.argsort(iou)[::-1]
        pick=0

        while pick < len(idx):
            mask_area = np.sum(masks[idx[pick] + i0]['segmentation'])
            # if mask is very large compared to size of the image (credit:segment everygrain) modified from 0.1 to 0.4
            if mask_area / (crop_size ** 2) <= size_threshold:
                break
            else:
                pick += 1
        if pick<len(idx):
            highest_pred_iou_by_point.append(masks[idx[pick]+i0])
        i+=1
    return highest_pred_iou_by_point

def Groupping_masks(list_of_masks):

    ar_masks=np.stack(list_of_masks)
    ar_masks_flat=ar_masks.reshape((ar_masks.shape[0],ar_masks.shape[1]*ar_masks.shape[2]))#flat 2d to 1d masks

    #find pixel-wise overlaps
    list_overlap=[]
    # Iterate through each column of the array
    for i in range(ar_masks_flat.shape[1]):
        # for each pixel find out the idex of mask where the pixel was in a mask
        nz = np.where(ar_masks_flat[:, i] != 0)[0]
        
        # if there are overlap of mask
        if len(nz) > 1:
            list_overlap.append(tuple(nz))

    #get uniqe pairs and intersection area(pixel) for each pair
    group_counter=Counter(list_overlap)
    unique_groups = [list(tup) for tup in group_counter.keys()]
    group_overlap_area = list(group_counter.values())
    return group_overlap_area, unique_groups, list_overlap

def filter_groupping_by_intersection(group_overlap_area,unique_groups, list_overlap ,intersection_threshold=1000):
    #filter by intersection area
    filtered=np.array(group_overlap_area)>intersection_threshold
    unique_groups_thresholded=[unique_groups[i] for i in range(len(unique_groups)) if filtered[i]]
    #report filter
    print(f'Threshold: {intersection_threshold} pixels, {len(list_overlap)-len(unique_groups_thresholded)} groups removed',
        f'\nOverlap groups before filtering: {len(list_overlap)}, after filtering: {len(unique_groups_thresholded)}')
    return unique_groups_thresholded

def checking_remaining_ungroupped(list_of_masks, unique_groups_thresholded):
    #check if there is remaining ungroupped pairs
    checker=np.zeros(len(list_of_masks))
    for gp in unique_groups_thresholded:
        for i in gp:
            checker[i]+=1
    remaining_ungroupped=np.max(np.unique(checker))

    cleaned_groups=None#dummy to identify first run

    while remaining_ungroupped>1:#keep groupping until there is no overlapping groups
        groups=[]
        if not cleaned_groups:#the first run
            temp_list_overlap=unique_groups_thresholded.copy()
        else:#>2 run
            temp_list_overlap=cleaned_groups.copy()

        while len(temp_list_overlap)>1:
            temp_group=temp_list_overlap[0].copy()
            remaining=[]
            for i in np.arange(1,len(temp_list_overlap),1):
                num_common_val=len(np.intersect1d(temp_list_overlap[0],temp_list_overlap[i]))
                if num_common_val!=0:
                    temp_group=np.hstack((temp_group,temp_list_overlap[i]))
                else:
                    remaining.append(temp_list_overlap[i])
            temp_list_overlap=remaining.copy()
            groups.append(temp_group)
        if len(remaining)>0:
            groups.append(remaining[0])

        #remove duplicated index after stacking
        cleaned_groups=[np.unique(group) for group in groups]

        checker=np.zeros(len(masks))
        for gp in cleaned_groups:
            for i in gp:
                checker[i]+=1
        remaining_ungroupped=np.max(np.unique(checker))
    if cleaned_groups:
        all_grouped_masks=np.unique(np.hstack(cleaned_groups))
        if len(all_grouped_masks)!=len(list_of_masks):
            list_of_nooverlap_mask=np.setdiff1d(np.arange(len(list_of_masks)), all_grouped_masks)
        else:
            list_of_nooverlap_mask=[]
    elif len(unique_groups_thresholded)>0:
        cleaned_groups=unique_groups_thresholded
        all_grouped_masks=np.unique(np.hstack(cleaned_groups))
        list_of_nooverlap_mask=np.setdiff1d(np.arange(len(list_of_masks)), all_grouped_masks)
        #list_of_nooverlap_mask=np.unique(np.hstack(unique_groups_thresholded))
    else:
        cleaned_groups=None
        list_of_nooverlap_mask=np.arange(len(list_of_masks))
    return cleaned_groups, list_of_nooverlap_mask

def Guided_second_pass_SAM(cleaned_groups,tm=0.5,ts=0.5):
    ##problem--we are assuming that in each disconnected region there is only one object
    cleaned_groups_reseg=[]
    for k in range(len(cleaned_groups)):
        stacked=np.stack([list_of_masks[i] for i in cleaned_groups[k]])
        mean_stacked=np.mean(stacked,axis=0)
        #std_stacked=np.std(stacked,axis=0)

        #separate high confidence region(high mean) and low
        labels=label(np.logical_and(mean_stacked<=tm,mean_stacked>0))
        regions=regionprops(labels)
        labels=label(np.logical_and(mean_stacked>tm,mean_stacked>0))
        regions_highmean=regionprops(labels)
        for region in regions_highmean:
            regions.append(region) 

        for props in regions:
            if props.area>100:#apply minimum area to filter out mini residuals
                y0, x0 = props.centroid
                input_point = np.array([[x0,y0]])
                input_label = np.array([1])

                partmasks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,)
                best_idx=np.argmax(scores)#pick the mask with highest score
                cleaned_groups_reseg.append({'mask':partmasks[best_idx],'score':scores[best_idx],'logit':logits[best_idx],'group':k})
    
    list_of_cleaned_groups_reseg_masks = [fnc.clean_mask(mask['mask'].astype('bool')) for mask in cleaned_groups_reseg]
    list_of_cleaned_groups_reseg_score=[mask['score'] for mask in cleaned_groups_reseg]
    return list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score

#containers
all_reseg=[]
for ij_idx in clipij:
    start_loop = time.time()
    print(f'Segmenting clip: {ij_idx}')
    ji=ij_idx[1]
    ii=ij_idx[0]
    if (ji*crop_size<(image.shape[1]) and ii*crop_size<(image.shape[0])):
        #prepare image
        pre_para={'Crop': {'crop size': crop_size, 'j':ji,'i':ii},
                #'Gaussian': {'kernel size':3}
                #'CLAHE':{'clip limit':2}#,
                #'Downsample': {'fxy':4},
                #'Buffering': {'crop size': crop_size}
                }
        temp_image=fnc.preprocessing_roulette(image, pre_para)
        if (temp_image.shape[0]>(crop_size//4) and temp_image.shape[1]>(crop_size//4)):
            if len(np.unique(temp_image))>1:
                #clear gpu ram
                gc.collect()
                torch.cuda.empty_cache()

                #SAM segmentation
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=pps,
                    pred_iou_thresh=0,
                    stability_score_thresh=0,#iou by varying cutoff in binary conversion
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
                print('First pass SAM: ', len(masks),' mask(s) found')

                #post processing
                #filter output mask per point by select highest pred iou mask
                masks=filter_by_pred_iou_and_size_per_seedpoint(masks)
                print('Filtered by highest predicted iou per seed point, ', len(masks),' mask(s) remains')

                list_of_pred_iou = [mask['predicted_iou'] for mask in masks]
                list_of_masks = [fnc.clean_mask(mask['segmentation'].astype('bool')) for mask in masks]#remove small disconnected parts
                no_area_after_cleaning=np.array([np.sum(mask)==0 for mask in list_of_masks])
                if np.any(no_area_after_cleaning):
                    list_of_masks = [mask for mask, keep in zip(list_of_masks, ~no_area_after_cleaning) if keep]
                    list_of_pred_iou = [iou for iou, keep in zip(list_of_pred_iou, ~no_area_after_cleaning) if keep]
                #remove background/edge mask
                flattened_rgb=np.sum(temp_image,axis=2)
                not_background_mask=np.array([np.any(flattened_rgb[mask.astype('bool')]>0) for mask in list_of_masks])
                if not np.all(not_background_mask):
                    list_of_masks = [mask for mask, keep in zip(list_of_masks, not_background_mask) if keep]
                    list_of_pred_iou = [mask for mask, keep in zip(list_of_pred_iou, not_background_mask) if keep]
                    print('Background masks removed')

                if len(list_of_masks)>0:
                    #grouping overlaps
                    group_overlap_area, unique_groups, list_overlap = Groupping_masks(list_of_masks)
                    unique_groups_thresholded = filter_groupping_by_intersection(group_overlap_area,unique_groups, list_overlap)
                    cleaned_groups, list_of_nooverlap_mask = checking_remaining_ungroupped(list_of_masks, unique_groups_thresholded)
                    if cleaned_groups:
                        list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score = Guided_second_pass_SAM(cleaned_groups)
                        if len(list_of_nooverlap_mask)>0:
                            for m in list_of_nooverlap_mask:
                                list_of_cleaned_groups_reseg_masks.append(list_of_masks[m].astype('bool'))
                                list_of_cleaned_groups_reseg_score.append(list_of_pred_iou[m])
                            list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms = fnc.nms(list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score)
                        print('Found ',len(list_of_cleaned_groups_reseg_score_nms), ' mask(s)/object(s) in the clip')
                    else:
                        list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score = list_of_masks, list_of_pred_iou
                        list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms = fnc.nms(list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score)
                    #saving outputs
                    all_reseg.append({#'mask':list_of_cleaned_groups_reseg_masks,
                                    'nms mask':list_of_cleaned_groups_reseg_masks_nms,
                                    #'mask pred iou':list_of_cleaned_groups_reseg_score,
                                    'nms mask pred iou': list_of_cleaned_groups_reseg_score_nms,
                                    'i':ii,
                                    'j':ji,
                                    'crop size':crop_size})
                else:
                    print('No valid mask were found')
            else:
                print('This crop is bacground/edge')
        else:
            print('This crop is too small')            
    else:
        print('Exceeded image boundary')
    end_loop = time.time()
    print('loop took: ', end_loop-start_loop)
np.save(OutDIR+'all_reseg_mask',np.hstack(all_reseg))
end_script = time.time()
print('script took: ', end_script-start_script)
print('First and second pass SAM completed. Output saved to '+OutDIR)