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

start_script = time.time()
#load image
OutDIR='/DATA/vito/output/testing_noplot2/'
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
#clips=[0,0.5,1,1.5,2]
#clipij=np.array(np.meshgrid(clips, clips)).T.reshape(-1,2)
downsample_factor=1
clipi=np.arange(0,(image.shape[0]*downsample_factor)//crop_size+1,0.5)
clipj=np.arange(0,(image.shape[1]*downsample_factor)//crop_size+1,0.5)
clipij=np.array(np.meshgrid(clipi, clipj)).T.reshape(-1,2)


#containers
all_sizes=[]
all_max_iou=[]
all_kdc_iou=[]
all_label_exclude_edge=[]
all_nearest_centroid_distance=[]
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
            print(len(masks))

            #post processing
            #filter output mask per point by select highest pred iou mask
            highest_pred_iou_by_point=[]
            for i in np.arange(0,len(masks),3):
                iou=[mask['predicted_iou'] for mask in masks[i:i+3]]
                highest_pred_iou_by_point.append(masks[np.argmax(iou)+i])
            masks=highest_pred_iou_by_point

            #grouping overlaps
            list_of_pred_iou=[mask['predicted_iou'] for mask in masks]
            list_of_stability_score=[mask['stability_score'] for mask in masks]
            list_of_masks = [fnc.clean_mask(mask['segmentation'].astype(np.uint8)) for mask in masks]#remove small disconnected parts
            #get centroid
            list_of_mask_centroid = [fnc.get_centroid(mask) for mask in list_of_masks]
            ar_masks=np.stack(list_of_masks)
            ar_masks_flat=ar_masks.reshape((ar_masks.shape[0],ar_masks.shape[1]*ar_masks.shape[2]))#flat 2d to 1d masks

            #find pixel wise overlaps
            list_overlap=[]
            # Iterate through each column of the array
            for i in range(ar_masks_flat.shape[1]):
                # for each pixel find out the idex of mask where the pixel was in a mask
                nz = np.where(ar_masks_flat[:, i] != 0)[0]
                
                # if there are overlap of mask
                if len(nz) > 1:
                    list_overlap.append(tuple(nz))

            #get uniqe pairs and intersection area(pixel) for each pair
            #list_overlap_intersection=[np.unique(np.sum([list_of_masks[i] for i in overlap],axis=0), return_counts=True)[1][-1] for overlap in list_overlap]
            group_counter=Counter(list_overlap)
            unique_groups = [list(tup) for tup in group_counter.keys()]
            group_overlap_area = list(group_counter.values())

            #filter by intersection area
            threshold=1000
            filtered=np.array(group_overlap_area)>threshold
            unique_groups_thresholded=[unique_groups[i] for i in range(len(unique_groups)) if filtered[i]]
            #list_overlap_threshold=[list_overlap[i] for i in range(len(list_overlap_intersection)) if list_overlap_intersection[i] > threshold]

            #report filter
            print(f'Threshold: {threshold} pixels, {len(list_overlap)-len(unique_groups_thresholded)} groups removed',
                f'\nOverlap groups before filtering: {len(list_overlap)}, after filtering: {len(unique_groups_thresholded)}')
            
            #check if there is remaining ungroupped pairs
            checker=np.zeros(len(masks))
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
            
            all_grouped_masks=np.unique(np.hstack(cleaned_groups))
            if len(all_grouped_masks)!=len(list_of_masks):
                list_of_nooverlap_mask=np.setdiff1d(np.arange(len(list_of_masks)), all_grouped_masks)
            else:
                list_of_nooverlap_mask=[]
            
            #apply to all groups guided resegmentation by overlaping confidence  
            tm=0.5
            ts=0.5
            cleaned_groups_reseg=[]
            for k in range(len(cleaned_groups)):
                stacked=np.stack([list_of_masks[i] for i in cleaned_groups[k]])
                mean_stacked=np.mean(stacked,axis=0)
                std_stacked=np.std(stacked,axis=0)
                
                #group_reseg=[]

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
            
            list_of_cleaned_groups_reseg_masks = [fnc.clean_mask(mask['mask'].astype(np.uint8)) for mask in cleaned_groups_reseg]
            list_of_cleaned_groups_reseg_score=[mask['score'] for mask in cleaned_groups_reseg]

            #adding back mask that does not overlap with any other mask
            for m in list_of_nooverlap_mask:
                if np.sum(list_of_masks[m])>threshold:
                    list_of_cleaned_groups_reseg_masks.append(list_of_masks[m].astype(np.uint8))
                    list_of_cleaned_groups_reseg_score.append(list_of_pred_iou[m])

            #NMS filtering
            bboxes = torch.tensor([fnc.find_bounding_boxes(mask) for mask in list_of_cleaned_groups_reseg_masks], device=DEVICE, dtype=torch.float)
            scores = torch.tensor(list_of_cleaned_groups_reseg_score, device=DEVICE, dtype=torch.float)
            labels = torch.zeros_like(bboxes[:, 0])

            keep = batched_nms(bboxes, scores, labels, 0.3)
            list_of_cleaned_groups_reseg_masks_nms=[list_of_cleaned_groups_reseg_masks[i] for i in keep]
            list_of_cleaned_groups_reseg_score_nms=[list_of_cleaned_groups_reseg_score[i] for i in keep]

            #saving outputs
            all_reseg.append({'mask':list_of_cleaned_groups_reseg_masks,
                            'nms mask':list_of_cleaned_groups_reseg_masks_nms,
                            'mask pred iou':list_of_cleaned_groups_reseg_score,
                            'nms mask pred iou': list_of_cleaned_groups_reseg_score_nms,
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


end_script = time.time()
print('script took: ', end_script-start_script)