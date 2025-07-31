import numpy as np
import glob
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import load_image, preprocessing_roulette, resample_fnc, get_centroid, update_mask_ious_shared

def accuracy(file_pth, shadow=False):
    '''
    Assess the accuracy of segmentation masks against ground truth masks.
    Parameters:
    - file_pth(str): Path to the directory containing the segmentation results.
    - shadow(bool): If True, the function will also consider shadows in the assessment.
    Returns:
    - None: The function saves the accuracy results in a .npy file in the original directory. 
    The npy file will contain:
        - 'point based': Point-based accuracy for each label.
        - 'iou': IoU for each label.
        - 'area': Area of each label.
        - 'segment area': Area of each segment in the segmentation mask.
        - 'segment hit': Whether the segment intersects with an actual label.
        - 'label_count': Number of unique labels in the ground truth mask.
        - 'mask_count': Number of unique segments in the segmentation mask.
        - 'number of layers': Indicates the number of layers used in the segmentation.
        - 'para': Parameters used for the segmentation.
        - 'completely_in_shadow': Indicates if the segment is completely in shadow (>90% covered). This is only included if shadow=True.
    '''
    with open(os.path.join(file_pth, 'para.json'), 'r') as json_file:
        para = json.load(json_file)[0]
    OutDIR=para.get('OutDIR')
    DataDIR=para.get('DataDIR')
    DSname=para.get('DatasetName')
    resample_factor=para.get('resample_factor')
    image=(np.load(os.path.join(DataDIR,DSname,'img.npy')))
    if resample_factor!=1:
        pre_para={'Resample': {'fxy':resample_factor},
            }

        image=preprocessing_roulette(image, pre_para)
        print('resampled to: ', image.shape)

    n_pass=len(os.listdir(OutDIR+'/Merged'))

    seg_masks=np.array(np.load(OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy', allow_pickle=True))

    third=n_pass
    print('Mask imported from '+OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy')
    print('masks size:', seg_masks.shape)
    print(len(np.unique(seg_masks)),' mask(s) loaded')


    mask=(np.load(os.path.join(DataDIR,DSname,'msk.npy'))).astype(np.uint16)
    if shadow:
        shadow_mask=(np.load(os.path.join(DataDIR,DSname,'shd.npy'))).astype(np.uint16)
    seg_masks_rs=resample_fnc(seg_masks.astype(np.uint16),{'target_size':mask.shape[::-1], 'method':'nearest'})
    print('No. of actual objects: '+str(len(np.unique(mask))-1))

    seg_ids=np.unique(seg_masks)
    centroids=[get_centroid(seg_masks==id) for id in seg_ids]
    centroids=np.array(centroids)/resample_factor

    ids, counts=np.unique(mask, return_counts=True)
    ids, counts = ids[1:], counts[1:]
    area = counts * (0.2 * 0.2)
    ids = ids[np.argsort(area)]
    area = np.sort(area)
    if shadow:
        shadowed_ids,shadowed_area=np.unique(mask[shadow_mask<1], return_counts=True)
        shadowed_ids, shadowed_area = shadowed_ids[1:], shadowed_area[1:]  # Exclude background
        shadowed_area = shadowed_area * (0.2 * 0.2)  # Convert to area in square mm
        completely_in_shadow=np.zeros_like(ids)
        for id in ids:
            if id in shadowed_ids:
                if shadowed_area[shadowed_ids==id]/area[ids==id]>0.9:
                    completely_in_shadow[ids==id]=1

    point_based_ac=np.zeros_like(ids)
    seg_fp= np.zeros_like(seg_ids)
    for c in range(len(centroids))[1:]:
        hit_id=int(mask[int(centroids[c][0]),int(centroids[c][1])])
        point_based_ac[ids==hit_id]+=1
        if hit_id!=0:
                seg_fp[c]+=1
    mask_ious = update_mask_ious_shared(centroids[1:], mask, ids, seg_masks_rs, seg_ids[1:])

    print('Mean mask IoU: ')
    print(np.mean(np.abs(mask_ious)))
    if shadow:
        np.save(os.path.join(DataDIR,DSname,file_pth,f'ac.npy')
                , {'point based':point_based_ac, 'iou':mask_ious
                   , 'area':area, 'segment area':np.unique(seg_masks,return_counts=True)[1][1:]/resample_factor
                   , 'segment hit':seg_fp[1:],'label_count':len(np.unique(mask))-1
                   ,'mask_count':len(np.unique(seg_masks)),'number of layers': third, 'para':para
                   , 'completely_in_shadow':completely_in_shadow})
    else:
        np.save(os.path.join(DataDIR,DSname,file_pth,f'ac.npy')
                , {'point based':point_based_ac, 'iou':mask_ious
                   , 'area':area, 'segment area':np.unique(seg_masks,return_counts=True)[1][1:]/resample_factor
                   , 'segment hit':seg_fp[1:],'label_count':len(np.unique(mask))-1
                   ,'mask_count':len(np.unique(seg_masks)),'number of layers': third, 'para':para})