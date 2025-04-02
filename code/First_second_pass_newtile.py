import torch
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator_mod2 as SamAutomaticMaskGenerator
import numpy as np
import torch
import functions as fnc
import gc
import os
import time
import sys
import json
from contextlib import redirect_stdout
import First_second_fnc as FS_fnc
import os
start_script = time.time()




OutDIR=sys.argv[1]

n_pass=len(os.listdir(OutDIR+'Merged'))
if not os.path.exists(OutDIR+f'chunks/{n_pass}'):
    os.makedirs(OutDIR+f'chunks/{n_pass}')
with open(OutDIR+'init_para.json', 'r') as json_file:
    init_para = json.load(json_file)[n_pass]
print('Loaded parameters from json')
print(init_para)


#OutDIR=init_para.get('OutDIR')
DataDIR=init_para.get('DataDIR')
DSname=init_para.get('DatasetName')
fid=init_para.get('fid')
pps=init_para.get('point_per_side')
b=init_para.get('b')
stb_t=init_para.get('stability_t')
#defining clips
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')
min_pixel=(init_para.get('expected_min_size(sqmm)')/(init_para.get('resolution(mm)')**2))*resample_factor
min_radi=init_para.get('min_radius')
print(f'Minimum expected size: {min_pixel} pixel')


try:#attempt to load saved pre_para
    with open(OutDIR+'pre_para.json', 'r') as json_file:
        pre_para = json.load(json_file)[n_pass]
    pre_para.update({'Resample': {'fxy':resample_factor}})
    print('Loaded preprocessing parameters from json')
    print(pre_para)
except:#use defined init_para
    print('Using preprocessing default')
    pre_para={'Resample': {'fxy':resample_factor}}
    print(pre_para)

#Prep
#load image
image=fnc.load_image(DataDIR,DSname,fid)
print('Image size:', image.shape)
image=fnc.preprocessing_roulette(image, pre_para)

patches = fnc.get_image_patches(image, crop_size, 2*b)
print(f'Tiled into {len(patches)} patches')
patch_keys=patches.keys()
max_ij=np.max(np.array(list(patch_keys)),axis=0)

#setup SAM
MODEL_TYPE = "vit_h"
DEVICE, CHECKPOINT_PATH=fnc.set_sam(MODEL_TYPE,DataDIR)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)


with open(OutDIR+f'chunks/{n_pass}/output.txt', 'w') as f:
    with redirect_stdout(f):
        for ij_idx in patch_keys:
            start_loop = time.time()
            print(f'Segmenting clip: {ij_idx}')
            #prepare image
            temp_image=patches[ij_idx]
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
                        stability_score_thresh=stb_t,#iou by varying cutoff in binary conversion
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
                    masks=FS_fnc.filter_by_pred_iou_and_size_per_seedpoint(masks, crop_size)
                    print('Filtered by highest predicted iou per seed point, ', len(masks),' mask(s) remains')

                    list_of_pred_iou = [mask['predicted_iou'] for mask in masks]
                    list_of_masks = [fnc.clean_mask(mask['segmentation'].astype('bool')) for mask in masks]#remove small disconnected parts
                    no_area_after_cleaning=np.array([np.sum(mask)==0 for mask in list_of_masks])
                    area_radi=np.array([fnc.area_radi(mask, min_pixel, min_radi) for mask in list_of_masks])
                    if np.any(no_area_after_cleaning):
                        list_of_masks = [mask for mask, keep in zip(list_of_masks, ~no_area_after_cleaning) if keep]
                        list_of_pred_iou = [iou for iou, keep in zip(list_of_pred_iou, ~no_area_after_cleaning) if keep]
                    if not np.all(area_radi):
                        list_of_masks = [mask for mask, keep in zip(list_of_masks, area_radi) if keep]
                        list_of_pred_iou = [iou for iou, keep in zip(list_of_pred_iou, area_radi) if keep]
                    #remove background/edge mask
                    flattened_rgb=np.sum(temp_image,axis=2)
                    not_background_mask=np.array([np.any(flattened_rgb[mask.astype('bool')]>0) for mask in list_of_masks])
                    if not np.all(not_background_mask):
                        list_of_masks = [mask for mask, keep in zip(list_of_masks, not_background_mask) if keep]
                        list_of_pred_iou = [mask for mask, keep in zip(list_of_pred_iou, not_background_mask) if keep]
                        print('Background masks removed')
                    
                    if len(list_of_masks)>0:
                        #grouping overlaps
                        list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms=[],[]
                        group_overlap_area, unique_groups, list_overlap = FS_fnc.Groupping_masks(list_of_masks)
                        unique_groups_thresholded = FS_fnc.filter_groupping_by_intersection(group_overlap_area,unique_groups, list_overlap)
                        cleaned_groups, list_of_nooverlap_mask = FS_fnc.checking_remaining_ungroupped(list_of_masks, unique_groups_thresholded, masks)
                        if cleaned_groups:
                            list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score = FS_fnc.Guided_second_pass_SAM(cleaned_groups, min_pixel, min_radi, list_of_masks, predictor, crop_size)
                            if len(list_of_nooverlap_mask)>0:
                                for m in list_of_nooverlap_mask:
                                    list_of_cleaned_groups_reseg_masks.append(list_of_masks[m].astype('bool'))
                                    list_of_cleaned_groups_reseg_score.append(list_of_pred_iou[m])
                            list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms = fnc.nms(list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score)
                            print('Found ',len(list_of_cleaned_groups_reseg_score_nms), ' mask(s)/object(s) in the clip')
                        else:
                            list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score = list_of_masks, list_of_pred_iou
                            list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms = fnc.nms(list_of_cleaned_groups_reseg_masks, list_of_cleaned_groups_reseg_score)
                            print(f'No groups were found, found {len(list_of_cleaned_groups_reseg_masks)} mask(s) from the first pass')
                            print(f'{len(list_of_cleaned_groups_reseg_masks_nms)} left after nms filtering')

                        #valid box
                        if len(list_of_cleaned_groups_reseg_masks_nms)>0:
                            keep = FS_fnc.mask_in_valid_box(list_of_cleaned_groups_reseg_masks_nms,b, ij_idx, max_ij)
                            list_of_cleaned_groups_reseg_masks_nms=[list_of_cleaned_groups_reseg_masks_nms[i] for i,k in enumerate(keep) if k]
                            list_of_cleaned_groups_reseg_score_nms=[list_of_cleaned_groups_reseg_score_nms[i] for i,k in enumerate(keep) if k]
                            if len(list_of_cleaned_groups_reseg_masks_nms)>0:
                                msk_dic={#'mask':list_of_cleaned_groups_reseg_masks,
                                            'nms mask':list_of_cleaned_groups_reseg_masks_nms,
                                            #'mask pred iou':list_of_cleaned_groups_reseg_score,
                                            'nms mask pred iou': list_of_cleaned_groups_reseg_score_nms,
                                            'ij':ij_idx,'crop size':crop_size}
                                np.save(OutDIR+f'chunks/{n_pass}/chunk_{int(ij_idx[0])}_{int(ij_idx[1])}',[msk_dic])
                                del msk_dic, list_of_cleaned_groups_reseg_masks_nms, list_of_cleaned_groups_reseg_score_nms
                        else:
                            print('No valid mask were found inside valid box')
                    else:
                        print('No valid mask remains after background, area, and radius filtering')
                else:
                    print('This crop is bacground/edge')
            else:
                print('This crop is too small')            
            end_loop = time.time()
            print('loop took: ', end_loop-start_loop)

#from scipy.stats import gaussian_kde
#plt.figure(figsize=(16, 10))
#plt.subplot(2,2,1)
#plt.xscale('log')
#data = stats['area']

#kde = gaussian_kde(data)
#x = np.linspace(min(data), max(data), 1000)
#kde_values = kde(x)
#plt.plot(x, kde_values)

#plt.xlabel('Area (pixel)')
#plt.ylabel('Density')
#plt.title('Density Plot of Area')
#plt.grid()

#plt.subplot(2,2,2)
#plt.xscale('log')
#frequencies, bin_edges = np.histogram(data, bins=30)
#bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
#plt.plot(bin_midpoints, frequencies)

#plt.xlabel('Area (pixel)')
#plt.ylabel('Frequency')
#plt.title('Frequency Plot of Area')
#plt.grid()

#plt.subplot(2,2,3)
#kde = gaussian_kde(data)
#x = np.linspace(min(data), max(data), 1000)
#kde_values = kde(x)
#plt.plot(x, kde_values)

#plt.xlabel('Area (pixel)')
#plt.ylabel('Density')
#plt.title('Density Plot of Area (nms)')
#plt.grid()

#plt.subplot(2,2,4)
#frequencies, bin_edges = np.histogram(data, bins=30)
#bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
#plt.plot(bin_midpoints, frequencies)

#plt.xlabel('Area (pixel)')
#plt.ylabel('Frequency')
#plt.title('Frequency Plot of Area (nms)')
#plt.grid()
#plt.suptitle(f'{ len(stats)} object(s)')
#plt.tight_layout()
#plt.savefig(OutDIR+'size_distribution.png')
#plt.show()

end_script = time.time()
print('script took: ', end_script-start_script)
print('First and second pass SAM completed. Output saved to '+OutDIR)