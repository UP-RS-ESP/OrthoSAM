import numpy as np
import functions as fnc
from skimage.measure import label, regionprops
from collections import Counter

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
    print(f'Threshold: {intersection_threshold} pixels, {len(list_overlap)-len(unique_groups_thresholded)} groups removed',
        f'\nOverlap groups before filtering: {len(list_overlap)}, after filtering: {len(unique_groups_thresholded)}')
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

def Guided_second_pass_SAM(cleaned_groups, min_pixel, min_radi, list_of_masks, predictor, crop_size, size_threshold=0.4,tm=0.5):
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
            if (props.area>min_pixel):#apply minimum area to filter out mini residuals
                y0, x0 = props.centroid
                input_point = np.array([[x0,y0]])
                input_label = np.array([1])

                partmasks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,)
                #best_idx=np.argmax(scores)#pick the mask with highest score
                partmasks=partmasks[np.argsort(scores)[::-1]]
                logits=logits[np.argsort(scores)[::-1]]
                scores=np.sort(scores)[::-1]
                pick=0
                while pick < len(partmasks):
                    mask_area = np.sum(partmasks[pick])
                    # if mask is very large compared to size of the image (credit:segment everygrain) modified from 0.1 to 0.4
                    if mask_area / (crop_size ** 2) <= size_threshold:
                        if fnc.area_radi(partmasks[pick], min_pixel, min_radi):
                            cleaned_groups_reseg.append({'mask':partmasks[pick],'score':scores[pick],'logit':logits[pick],'seed_point':props.centroid})
                            break
                        else:
                            pick += 1
                    else:
                        pick += 1
                
    
    list_of_cleaned_groups_reseg_masks = [fnc.clean_mask(mask['mask'].astype('bool')) for mask in cleaned_groups_reseg]
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