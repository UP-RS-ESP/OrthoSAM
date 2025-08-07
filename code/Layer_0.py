import torch
from segment_anything import SamPredictor
from automatic_mask_generator_mod import SamAutomaticMaskGenerator
import numpy as np
import torch
from skimage.measure import label, regionprops
from collections import Counter
import gc
import os
import time
import json
import os
from utility import load_image, preprocessing_roulette, get_image_patches, set_sam, nms, clean_mask, area_radi

def predict_tiles(para_list,n_pass): 
    start_script = time.time()
    
    para = para_list[n_pass]
    OutDIR=para.get('OutDIR')
    
    #try:
    if n_pass==0:
        print(f'---------------\nLayer {n_pass}')
    print('\tSegment tiles\n\n')

    if not os.path.exists(os.path.join(OutDIR,f'chunks/{n_pass}')):
        os.makedirs(os.path.join(OutDIR,f'chunks/{n_pass}'))

    
    print('\tLoaded parameters from json')
    print('\t',para)

    DataDIR=para.get('DataDIR')
    DSname=para.get('DatasetName')
    fid=para.get('fid')
    pps=para.get('input_point_per_axis')
    b=para.get('tile_overlap')
    stb_t=para.get('stability_t')
    #defining clips
    crop_size=para.get('tile_size')
    resample_factor=para.get('resample_factor')
    min_pixel=(para.get('expected_min_size(sqmm)')/(para.get('resolution(mm)')**2))*resample_factor
    min_radi=para.get('min_radius')
    print(f'\tMinimum expected size: {min_pixel} pixel')


    try:#attempt to load saved pre_para
        with open(os.path.join(OutDIR,'pre_para.json'), 'r') as json_file:
            pre_para = json.load(json_file)[n_pass]
        pre_para.update({'Resample': {'fxy':resample_factor}})
        print('\tLoaded preprocessing parameters from json')
        print('\t',pre_para)
    except:#use defined para
        print('\tNo pre_para found. Only applying resampling.')
        pre_para={'Resample': {'fxy':resample_factor}}
        print('\t',pre_para)

    #Preprocessing
    #load image
    image=load_image(DataDIR,DSname,fid)
    print('\tImage size:', image.shape)
    image=preprocessing_roulette(image, pre_para)

    patches = get_image_patches(image, crop_size, b)
    print(f'\tTiled into {len(patches)} tiles')
    patch_keys=patches.keys()
    max_ij=np.max(np.array(list(patch_keys)),axis=0)

    #setup SAM
    sam=set_sam(para.get('MODEL_TYPE'), para.get('CheckpointDIR'))

    mask_generator = SamAutomaticMaskGenerator(sam)

    for count,ij_idx in enumerate(patch_keys):
        start_loop = time.time()
        print(f'\tSegmenting tile: {ij_idx} [{count+1}/{len(patch_keys)}]')
        #prepare image
        temp_image=patches[ij_idx]
        if (temp_image.shape[0]>(crop_size//8) and temp_image.shape[1]>(crop_size//8)):
            if len(np.unique(temp_image))>1:
                #clear gpu ram
                gc.collect()
                torch.cuda.empty_cache()

                #SAM segmentation
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=pps,
                    pred_iou_thresh=0,
                    stability_score_thresh=stb_t,#iou by varying cutoff in binary conversion
                    box_nms_thresh=0.3,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks
                    crop_n_layers=0,#cut into 2**n crops
                    crop_nms_thresh=0,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks between crops
                    crop_n_points_downscale_factor=1,
                    crop_overlap_ratio=0,
                    #min_mask_region_area=2000,
                )

                with torch.no_grad():
                    masks = mask_generator.generate(temp_image)
                print(f'\tFirst pass SAM: {len(masks)} mask(s) found')

                #post processing
                #filter output mask per point by select highest pred iou mask
                masks=filter_by_pred_iou_and_size_per_seedpoint(masks, crop_size)
                print(f'\tFiltered by highest predicted iou per seed point, {len(masks)} mask(s) remains')

                list_of_pred_iou = [mask['predicted_iou'] for mask in masks]
                list_of_masks = [clean_mask(mask['segmentation'].astype('bool')) for mask in masks]#remove small disconnected parts
                no_area_after_cleaning=np.array([np.sum(mask)==0 for mask in list_of_masks])
                area_radi_flt=np.array([area_radi(mask, min_pixel, min_radi) for mask in list_of_masks])
                if np.any(no_area_after_cleaning):
                    list_of_masks = [mask for mask, keep in zip(list_of_masks, ~no_area_after_cleaning) if keep]
                    list_of_pred_iou = [iou for iou, keep in zip(list_of_pred_iou, ~no_area_after_cleaning) if keep]
                if not np.all(area_radi_flt):
                    list_of_masks = [mask for mask, keep in zip(list_of_masks, area_radi_flt) if keep]
                    list_of_pred_iou = [iou for iou, keep in zip(list_of_pred_iou, area_radi_flt) if keep]
                #remove background/edge mask
                flattened_rgb=np.sum(temp_image,axis=2)
                not_background_mask=np.array([np.any(flattened_rgb[mask.astype('bool')]>0) for mask in list_of_masks])
                if not np.all(not_background_mask):
                    list_of_masks = [mask for mask, keep in zip(list_of_masks, not_background_mask) if keep]
                    list_of_pred_iou = [mask for mask, keep in zip(list_of_pred_iou, not_background_mask) if keep]
                    print('\tBackground masks removed')
                
                if len(list_of_masks)>0:
                    #grouping overlaps
                    list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms=[],[]
                    group_overlap_area, unique_groups, list_overlap = Groupping_masks(list_of_masks)
                    unique_groups_thresholded = filter_groupping_by_intersection(group_overlap_area,unique_groups, list_overlap)
                    cleaned_groups, list_of_nooverlap_mask = checking_remaining_ungroupped(list_of_masks, unique_groups_thresholded, masks)
                    if cleaned_groups:
                        predictor = SamPredictor(sam)
                        predictor.set_image(temp_image)
                        list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score = Guided_second_pass_SAM(cleaned_groups, min_pixel, min_radi, list_of_masks, predictor, crop_size,stb_t)
                        if len(list_of_nooverlap_mask)>0:
                            for m in list_of_nooverlap_mask:
                                list_of_cleaned_groups_reseg_masks.append(list_of_masks[m].astype('bool'))
                                list_of_cleaned_groups_reseg_score.append(list_of_pred_iou[m])
                        list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms = nms(list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score)
                        print(f'\tFound {len(list_of_cleaned_groups_reseg_score_nms)} mask(s)/object(s) in the tile')
                    else:
                        list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score = list_of_masks, list_of_pred_iou
                        list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms = nms(list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score)
                        print(f'\tNo groups were found, found {len(list_of_cleaned_groups_reseg_masks)} mask(s) from the first pass')
                        print(f'\t{len(list_of_cleaned_groups_reseg_masks_nms)} left after nms filtering')

                    #valid box
                    if len(list_of_cleaned_groups_reseg_masks_nms)>0:
                        keep = mask_in_valid_box(list_of_cleaned_groups_reseg_masks_nms,b//2, ij_idx, max_ij)
                        list_of_cleaned_groups_reseg_masks_nms=[list_of_cleaned_groups_reseg_masks_nms[i] for i,k in enumerate(keep) if k]
                        list_of_cleaned_groups_reseg_score_nms=[list_of_cleaned_groups_reseg_score_nms[i] for i,k in enumerate(keep) if k]
                        if len(list_of_cleaned_groups_reseg_masks_nms)>0:
                            msk_dic={#'mask':list_of_cleaned_groups_reseg_masks,
                                        'nms mask':list_of_cleaned_groups_reseg_masks_nms,
                                        #'mask pred iou':list_of_cleaned_groups_reseg_score,
                                        'nms mask pred iou': list_of_cleaned_groups_reseg_score_nms,
                                        'ij':ij_idx,'crop size':crop_size}
                            np.save(os.path.join(OutDIR,f'chunks/{n_pass}/chunk_{int(ij_idx[0])}_{int(ij_idx[1])}'),[msk_dic])
                            del msk_dic, list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms
                    else:
                        print('\tNo valid mask were found inside valid box')
                else:
                    print('\tNo valid mask remains after background, area, and radius filtering')
            else:
                print('\tThis crop is bacground/edge')
        else:
            print('\tThis crop is too small')            
        end_loop = time.time()
        print(f'\tloop took: {end_loop-start_loop:.2f} seconds')


    end_script = time.time()
    print(f'\tscript took: {end_script-start_script:.2f} seconds')
    print(f'\tLayer {n_pass} segment tiles completed. Output saved to '+OutDIR)
    print('---------------\n\n\n\n\n\n')
    #except Exception as e:
    #    import traceback
    #    traceback.print_exc()



def filter_by_pred_iou_and_size_per_seedpoint(masks,crop_size,size_threshold=0.4):
    seed_point=np.array([mask['point_coords'][0] for mask in masks])
    highest_pred_iou_by_point=[]
    i=0
    while i < len(masks):
        i0=i
        if (i+2)<len(masks):
            if np.all(seed_point[i]==seed_point[i+2]):
                i+=2
            elif np.all(seed_point[i]==seed_point[i+1]):
                i+=1
        elif (i+1)<len(masks):
            if np.all(seed_point[i]==seed_point[i+1]):
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
    print(f'\tThreshold: {intersection_threshold} pixels, {len(list_overlap)-len(unique_groups_thresholded)} groups removed',
        f'\n\tOverlap groups before filtering: {len(list_overlap)}, after filtering: {len(unique_groups_thresholded)}')
    return unique_groups_thresholded

def checking_remaining_ungroupped(list_of_masks, unique_groups_thresholded, masks):
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

def calculate_stability_score(
    mask, mask_threshold: float, threshold_offset: float
):
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    high_mask = mask > (mask_threshold + threshold_offset)
    low_mask = mask > (mask_threshold - threshold_offset)

    intersection = np.logical_and(high_mask, low_mask).sum()
    union = np.logical_or(high_mask, low_mask).sum()

    if union == 0:
        return 0.0
    return intersection / union

def Guided_second_pass_SAM(cleaned_groups, min_pixel, min_radi, list_of_masks, predictor, crop_size,stb_t, size_threshold=0.4,tm=0.5):
   
    cleaned_groups_reseg=[]
    for k in range(len(cleaned_groups)):
        stacked=np.stack([list_of_masks[i] for i in cleaned_groups[k]])
        mean_stacked=np.mean(stacked,axis=0)

        #separate high confidence region(high mean) and low
        labels=label(np.logical_and(mean_stacked<=tm,mean_stacked>0))
        regions=regionprops(labels)
        labels=label(np.logical_and(mean_stacked>tm,mean_stacked>0))
        regions_highmean=regionprops(labels)
        for region in regions_highmean:
            regions.append(region) 

        for props in regions:
            if (props.area>min_pixel):#apply minimum area to filter out mini residuals
                y0, x0 = props.centroid
                input_point = np.array([[x0,y0]])
                input_label = np.array([1])

                partmasks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                    return_logits=True,
                    )
                #stability_score = calculate_stability_score(partmasks, 0, 1)
                stability_score=np.array([calculate_stability_score(mask, 0, 1) for mask in partmasks])
                partmasks=partmasks[np.argsort(scores)[::-1]]
                stability_score=stability_score[np.argsort(scores)[::-1]]
                scores=np.sort(scores)[::-1]
                
                pick=0
                while pick < len(partmasks):
                    mask_area = np.sum(partmasks[pick]>0)
                    # if mask is very large compared to size of the image (credit:segment everygrain) modified from 0.1 to 0.4
                    if mask_area / (crop_size ** 2) <= size_threshold:
                        if area_radi(partmasks[pick]>0, min_pixel, min_radi):
                            if stability_score[pick]>0.9:
                                cleaned_groups_reseg.append({'mask':partmasks[pick]>0,'score':scores[pick],'seed_point':props.centroid})
                                break
                            else:
                                pick += 1
                        else:
                            pick += 1
                    else:
                        pick += 1
                
    
    list_of_cleaned_groups_reseg_masks = [clean_mask(mask['mask'].astype('bool')) for mask in cleaned_groups_reseg]
    list_of_cleaned_groups_reseg_score=[mask['score'] for mask in cleaned_groups_reseg]
    return list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score

def mask_in_valid_box(masks,b, ij_idx, max_ij):
    def get_box(b, ij_idx, max_ij):
        if (ij_idx[0]==max_ij[0] and ij_idx[0]==0):
            y0,y1=0,-1
        elif ij_idx[0]==max_ij[0]:
            y0,y1=b,-1
        elif ij_idx[0]==0:
            y0,y1=0,-b
        else:
            y0,y1=b,-b

        if (ij_idx[1]==max_ij[1] and ij_idx[1]==0):
            x0,x1=0,-1
        elif ij_idx[1]==max_ij[1]:#if it is the last tile, valid box should cover the entire 
            x0,x1=b,-1
        elif ij_idx[1]==0:
            x0,x1=0,-b
        else:
            x0,x1=b,-b
        return x0,x1,y0,y1
    x0,x1,y0,y1 = get_box(b, ij_idx, max_ij)
    keep=[]
    for mask in masks:
        area_in_box=np.sum(mask[y0:y1,x0:x1])
        total_area=np.sum(mask)
        if (total_area-area_in_box)==0:
            keep.append(True)
        elif (total_area-area_in_box)<area_in_box:
            keep.append(True)
        else:
            keep.append(False)
    return keep