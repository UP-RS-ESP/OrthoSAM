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
OutDIR='/DATA/vito/output/overlap_highestIOU_reseg_edgenobuffer_2/'
if not os.path.exists(OutDIR[:-1]):
    os.makedirs(OutDIR[:-1])
DataDIR='/DATA/vito/data/'
#fn_img = glob.glob(DataDIR+'test_img/*')
fn_img = glob.glob(DataDIR+'drone_ortho/*')
fn_img.sort()

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
                            'crop size':crop_size,
                            'fxy':downsample_factor})
            np.save(OutDIR+'all_reseg_mask',np.hstack(all_reseg))
            #plt.figure(figsize=(10,10))
            #plt.suptitle(f'Clip: i{ii},j{ji}')
            #plt.subplot(2,2,1)
            #plt.imshow(temp_image)
            #plt.imshow(np.sum(list_of_cleaned_groups_reseg_masks_nms,axis=0),alpha=0.5)
            #plt.title(f'Number of mask: {len(list_of_cleaned_groups_reseg_masks_nms)}')
            #plt.axis('off')
            #plt.subplot(2,2,2)
            #plt.imshow(np.sum([list_of_cleaned_groups_reseg_masks_nms[i]*(i+1) for i in range(len(list_of_cleaned_groups_reseg_masks_nms))],axis=0), cmap='nipy_spectral')
            #plt.title(f'Max overlap: {np.max(np.sum(list_of_cleaned_groups_reseg_masks_nms,axis=0))}')
            #plt.axis('off');
            #plt.subplot(2,2,3)
            #plt.imshow(np.sum(list_of_cleaned_groups_reseg_masks,axis=0))
            #plt.axis('off')
            #plt.title(f're segmented, max overlap {np.max(np.sum(list_of_cleaned_groups_reseg_masks,axis=0))} masks');
            #plt.subplot(2,2,4)
            #plt.imshow(np.sum(list_of_cleaned_groups_reseg_masks_nms,axis=0))
            #plt.axis('off')
            #plt.title(f'filtered and re segmented, max overlap {np.max(np.sum(list_of_cleaned_groups_reseg_masks_nms,axis=0))} masks');
            #plt.tight_layout()
            #plt.savefig(OutDIR+f'clip_{ii}{ji}_overlay.png')
            #plt.show()

            # checking accuracy
            #load label
            #temp_seg_id=fnc.preprocessing_roulette(seg_ids, pre_para)
            temp_seg_id=seg_ids[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]
            #temp_seg_id=fnc.buffering_fnc(temp_seg_id,{'crop size': crop_size})

            seg_labels=(np.unique(temp_seg_id)[1:])

            seg_size=[]
            for seg in seg_labels:
                seg_size.append(np.sum(temp_seg_id==seg))

            seg_size_sort_idx=np.argsort(seg_size)

            list_of_label=[(temp_seg_id==seg_labels[index]).astype(np.uint8) for index in seg_size_sort_idx]
            list_of_label_centroid = [fnc.get_centroid(mask) for mask in list_of_label]
            ar_label=np.stack(list_of_label)

            list_of_mask_centroid = [fnc.get_centroid(mask) for mask in list_of_cleaned_groups_reseg_masks_nms]
            ar_masks=np.stack(list_of_cleaned_groups_reseg_masks_nms)

            ar_all=np.stack(list_of_label+list_of_cleaned_groups_reseg_masks_nms)
            list_of_centroid = [fnc.get_centroid(ar_all[i]) for i in range(len(ar_all))]

            tree = KDTree(list_of_centroid)              
            k=4
            _, ind = tree.query(list_of_centroid[:len(list_of_label)], k=k)

            # get nearest centroid iou
            kdc_iou=[]
            nearest_mask_pick=[]
            for match in ind:
                #check if nearest neighbor is not a label
                j=1
                while match[j]<len(list_of_label):
                    j+=1
                    if j==k:
                        break
                if j<k:
                    if np.sum(ar_all[match[0]])>np.sum(ar_all[match[j]]):
                        kdc_iou.append(-fnc.iou(ar_all[match[0]],ar_all[match[j]]))
                        nearest_mask_pick.append(match[j])
                    else:
                        kdc_iou.append(fnc.iou(ar_all[match[0]],ar_all[match[j]]))
                        nearest_mask_pick.append(match[j])
                else:
                    kdc_iou.append(-0.1)
                    nearest_mask_pick.append(0)
        
            plt.figure(figsize=(10,10))
            plt.suptitle(f'Clip: i{ii},j{ji}')
            plt.subplot(2,2,1)
            plt.imshow(np.sum([list_of_cleaned_groups_reseg_masks_nms[i]*(i+1) for i in range(len(list_of_cleaned_groups_reseg_masks_nms))],axis=0), cmap='nipy_spectral')
            plt.title(f'SAM {len(list_of_cleaned_groups_reseg_masks_nms)} objects Max overlap: {np.max(np.sum(list_of_cleaned_groups_reseg_masks_nms,axis=0))}')
            plt.axis('off')
            plt.subplot(2,2,2)
            plt.imshow(np.sum([list_of_label[i]*(i+1) for i in range(len(list_of_label))],axis=0), cmap='nipy_spectral')
            plt.title(f'Label {len(list_of_label)} objects Max overlap: {len(np.unique(list_of_label))}')
            plt.axis('off')
            plt.subplot(2,2,3)
            plt.imshow(np.sum(list_of_label,axis=0).astype(np.int64)-np.sum(list_of_cleaned_groups_reseg_masks_nms,axis=0).astype(np.int64),cmap='seismic', vmax=2)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('Label SAM difference\nRed: only in label, Blue: only in SAM')
            plt.axis('off')

            plt.subplot(2,2,4)
            ar_label_iou=ar_label.copy().astype(np.float64)
            for i,x in enumerate(kdc_iou):
                ar_label_iou[i]=ar_label[i]*x
            #ar_label_iou_abs_sum=np.abs(np.sum(ar_label_iou,axis=0))
            #ar_label_iou_abs_sum[np.sum(ar_label,axis=0)==0]=-999
            #plt.imshow(ar_label_iou_abs_sum, cmap='seismic',vmax=1,vmin=0)
            mask = np.isclose(np.sum(ar_label,axis=0),0)
            plt.imshow(np.abs(np.sum(ar_label_iou,axis=0)), cmap='seismic',vmax=1,vmin=0)
            plt.colorbar(label='iou',fraction=0.046, pad=0.04)
            cmap = mcolors.ListedColormap(['none', 'yellow'])
            plt.imshow(mask, cmap=cmap, alpha=0.6)
            plt.axis('off')
            plt.title('Labels colored by nearest neighbour iou\nBlue: low iou, Yellow: background)')
            #plt.savefig(OutDIR+f'clip_{ii}{ji}_label_pred_compare.png')
            plt.show()

            nearest_centroid_distance=[]
            for i in range(len(list_of_label)):
                x0,y0=list_of_centroid[i]#label centroid
                x1,y1=list_of_centroid[nearest_mask_pick[i]]
                nearest_centroid_distance.append(np.sqrt((x0-x1)**2+(y0-y1)**2))
            
            ar_all_flat=ar_all.reshape((ar_all.shape[0],ar_all.shape[1]*ar_all.shape[2]))
            set_overlap = set()

            # Iterate through each column of the array
            for i in range(ar_all_flat.shape[1]):
                # for each pixel find out the idex of mask where the pixel was in a mask
                nz = np.where(ar_all_flat[:, i] != 0)[0]
                
                # if there are overlap of mask
                if len(nz) > 1:
                    nz_tuple = tuple(nz)
                    set_overlap.add(nz_tuple)

            # Convert the set back to a list if needed
            list_overlap = list(set_overlap)

            # get all average and max iou
            overlap_by_label=[]
            mask_label_inter=[]
            max_iou=[]
            avg_iou=[]
            max_mask_pick=[]
            for i in range(len(list_of_label)):
                all_pairs=[tup for tup in list_overlap if i in tup]
                if len(all_pairs)>0:
                    overlap_with_i=np.unique(np.hstack(all_pairs))[1:]
                    overlap_by_label.append(overlap_with_i)
                    agg_iou=[]
                    agg_inter=[]
                    for j in overlap_with_i:
                        if np.sum(ar_all[i])>np.sum(ar_all[j]):
                            agg_iou.append(-fnc.iou(ar_all[i],ar_all[j]))
                        else:
                            agg_iou.append(fnc.iou(ar_all[i],ar_all[j]))
                        agg_inter.append(np.logical_and(ar_all[i], ar_all[j]).sum())
                    mask_label_inter.append(agg_inter)
                    avg_iou.append(np.mean(agg_iou))
                    max_iou.append(agg_iou[np.argmax(np.absolute(agg_iou))])
                    max_mask_pick.append(overlap_with_i[np.argmax(np.absolute(agg_iou))])
                else:
                    overlap_by_label.append(np.array([]))
                    mask_label_inter.append(np.array([]))
                    avg_iou.append(0)
                    max_iou.append(0)
            
            #nearest_subset_masks=[list_of_cleaned_groups_reseg_masks_nms[i-len(list_of_label)] for i in nearest_mask_pick]
            #max_subset_masks=[list_of_cleaned_groups_reseg_masks_nms[i-len(list_of_label)] for i in max_mask_pick]

            plt.figure(figsize=(20,15))
            plt.suptitle(f'Clip: i{ii},j{ji}')
            plt.subplot(1,3,1)
            ar_label_dist=ar_label.copy()
            for i,x in enumerate(nearest_centroid_distance):
                ar_label_dist[i]=ar_label_dist[i]*x
            plt.imshow(np.sum(ar_label_dist,axis=0))
            plt.colorbar(label='pixel',fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.title('Colored by centroid distance')

            plt.subplot(1,3,2)
            ar_label_iou=ar_label.copy().astype(np.float64)
            for i,x in enumerate(kdc_iou):
                ar_label_iou[i]=ar_label[i]*x
            plt.imshow(np.sum(ar_label_iou,axis=0), cmap='seismic',vmax=1,vmin=-1)
            plt.colorbar(label='iou',fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.title('Colored by nearest neighbour signed iou')
            plt.savefig(OutDIR+f'clip_{ii}{ji}_dist_iou_overlay.png')
            plt.show()

            max_iou=np.array(max_iou)
            kdc_iou=np.array(kdc_iou)
            sizes=np.log10([seg_size[i] for i in seg_size_sort_idx])

            fig,ax=plt.subplots(2,1,figsize=(20,15))
            ax=ax.flatten()
            #ax[0].plot(np.log10([seg_size[i] for i in seg_size_sort_idx]),avg_iou,label='average iou')
            #ax[0].plot(np.log10([seg_size[i] for i in seg_size_sort_idx]),max_iou,label=f'max iou: {np.mean(max_iou[max_iou>0])}')
            #ax[0].plot(np.log10([seg_size[i] for i in seg_size_sort_idx]),kdc_iou,label=f'nearest neighbour: {np.mean(kdc_iou[kdc_iou>0])}')
            ax[0].scatter(sizes,np.array(nearest_centroid_distance),label=f'centroid distance (mean: {np.mean(nearest_centroid_distance):.2f})')
            ax[0].set_ylim(top=150)
            ax[0].set_xlabel('grain size (log)', fontsize=20)
            ax[0].set_ylabel('distance (px)', fontsize=20)
            ax[0].grid()
            ax[0].legend()
            #ax[1].scatter(np.log10([seg_size[i] for i in seg_size_sort_idx]),avg_iou,label='average iou')
            ax[1].scatter(sizes,max_iou,label=f'max iou (abs mean: {round(np.mean(np.abs(max_iou)),2)})')
            ax[1].scatter(sizes,kdc_iou,label=f'nearest neighbour (mean: {round(np.mean(kdc_iou),2)})')
            ax[1].set_xlabel('grain size (log)', fontsize=20)
            ax[1].set_ylabel('iou', fontsize=20)
            ax[1].grid()
            ax[1].legend()
            ax[0].set_title(f'Clip: i{ii},j{ji} by grain size (log)', fontsize=20)
            #plt.savefig(OutDIR+f'clip_{ii}{ji}_dist_iou_plot.png')
            plt.show()

            labels_exclude_edge=[]
            labels_exclude_edge_idx=[]
            for i in range(len(ar_label)):
                if (np.any(ar_label[i][0]==1) or np.any(ar_label[i][-1]==1) or np.any(ar_label[i][:,0]==1) or np.any(ar_label[i][:,-1]==1)):
                    labels_exclude_edge.append(False)
                else:
                    labels_exclude_edge.append(True)
                    labels_exclude_edge_idx.append(i)

            fig,ax=plt.subplots(2,1,figsize=(20,15))
            ax=ax.flatten()
            mean=np.mean(np.array(nearest_centroid_distance)[labels_exclude_edge])
            ax[0].scatter(sizes[labels_exclude_edge],np.array(nearest_centroid_distance)[labels_exclude_edge]
                        ,label=f'centroid distance (mean: {mean:.2f})')
            ax[0].set_ylim(top=150)
            ax[0].set_xlabel('grain size (log)', fontsize=20)
            ax[0].set_ylabel('distance (px)', fontsize=20)
            ax[0].grid()
            ax[0].legend()
            #ax[1].scatter(np.log10([seg_size[i] for i in seg_size_sort_idx]),avg_iou,label='average iou')
            ax[1].scatter(sizes[labels_exclude_edge],max_iou[labels_exclude_edge]
                        ,label=f'max iou (abs mean: {round(np.mean(np.abs(max_iou)[labels_exclude_edge]),2)})')
            ax[1].scatter(sizes[labels_exclude_edge],kdc_iou[labels_exclude_edge]
                        ,label=f'nearest neighbour (mean: {round(np.mean(np.array(kdc_iou)[labels_exclude_edge]),2)})')
            ax[1].set_xlabel('grain size (log)', fontsize=20)
            ax[1].set_ylabel('iou', fontsize=20)
            ax[1].grid()
            ax[1].legend()
            ax[0].set_title(f'Clip: i{ii},j{ji} Excluding edge objects', fontsize=20)
            #plt.savefig(OutDIR+f'clip_{ii}{ji}_dist_iou_plot_noedge.png')
            plt.show()

            all_sizes.append(sizes)
            all_max_iou.append(max_iou)
            all_kdc_iou.append(kdc_iou)
            all_label_exclude_edge.append(labels_exclude_edge)
            all_nearest_centroid_distance.append(nearest_centroid_distance)

            np.save(OutDIR+'all_size',np.hstack(all_sizes))
            np.save(OutDIR+'all_max_iou',np.hstack(all_max_iou))
            np.save(OutDIR+'all_kdc_iou',np.hstack(all_kdc_iou))
            np.save(OutDIR+'all_label_exclude_edge',np.hstack(all_label_exclude_edge))
            np.save(OutDIR+'all_nearest_centroid_distance',np.hstack(all_nearest_centroid_distance))
            

        else:
            print('This crop is too small')
    else:
        print('Exceeded image boundary')
    end_loop = time.time()
    print('loop took: ', end_loop-start_loop)



np.save(OutDIR+'all_size',np.hstack(all_sizes))
np.save(OutDIR+'all_max_iou',np.hstack(all_max_iou))
np.save(OutDIR+'all_kdc_iou',np.hstack(all_kdc_iou))
np.save(OutDIR+'all_label_exclude_edge',np.hstack(all_label_exclude_edge))
np.save(OutDIR+'all_nearest_centroid_distance',np.hstack(all_nearest_centroid_distance))
#np.save(OutDIR+'all_reseg_mask',np.hstack(all_reseg))

plt_max_iou=np.hstack(all_max_iou)
plt_kdc_iou=np.hstack(all_kdc_iou)
plt_sizes=np.hstack(all_sizes)
plt_nearest_centroid_distance=np.hstack(all_nearest_centroid_distance)
plt_labels_exclude_edge=np.hstack(all_label_exclude_edge)

fig,ax=plt.subplots(2,1,figsize=(20,15))
ax=ax.flatten()
#ax[0].plot(np.log10([seg_size[i] for i in seg_size_sort_idx]),avg_iou,label='average iou')
#ax[0].plot(np.log10([seg_size[i] for i in seg_size_sort_idx]),max_iou,label=f'max iou: {np.mean(max_iou[max_iou>0])}')
#ax[0].plot(np.log10([seg_size[i] for i in seg_size_sort_idx]),kdc_iou,label=f'nearest neighbour: {np.mean(kdc_iou[kdc_iou>0])}')
ax[0].scatter(plt_sizes,plt_nearest_centroid_distance,label=f'centroid distance (mean: {np.mean(plt_nearest_centroid_distance):.2f})')
ax[0].set_ylim(top=150)
ax[0].set_xlabel('grain size (log)', fontsize=20)
ax[0].set_ylabel('distance (px)', fontsize=20)
ax[0].grid()
ax[0].legend()
#ax[1].scatter(np.log10([seg_size[i] for i in seg_size_sort_idx]),avg_iou,label='average iou')
ax[1].scatter(plt_sizes,plt_max_iou,label=f'max iou (abs mean: {round(np.mean(np.abs(plt_max_iou)),2)})')
ax[1].scatter(plt_sizes,plt_kdc_iou,label=f'nearest neighbour (mean: {round(np.mean(plt_kdc_iou),2)})')
ax[1].set_xlabel('grain size (log)', fontsize=20)
ax[1].set_ylabel('iou', fontsize=20)
ax[1].grid()
ax[1].legend()
ax[0].set_title(f'by grain size (log), no. clips: {len(all_label_exclude_edge)}, no. samples: {len(plt_labels_exclude_edge)}', fontsize=20)
plt.savefig(OutDIR+f'all_dist_iou_plot.png')
plt.show()



fig,ax=plt.subplots(2,1,figsize=(20,15))
ax=ax.flatten()
mean=np.mean(np.array(plt_nearest_centroid_distance)[plt_labels_exclude_edge])
ax[0].scatter(plt_sizes[plt_labels_exclude_edge],np.array(plt_nearest_centroid_distance)[plt_labels_exclude_edge]
              ,label=f'centroid distance (mean: {mean:.2f})')
ax[0].set_ylim(top=150)
ax[0].set_xlabel('grain size (log)', fontsize=20)
ax[0].set_ylabel('distance (px)', fontsize=20)
ax[0].grid()
ax[0].legend()
#ax[1].scatter(np.log10([seg_size[i] for i in seg_size_sort_idx]),avg_iou,label='average iou')
ax[1].scatter(plt_sizes[plt_labels_exclude_edge],plt_max_iou[plt_labels_exclude_edge]
              ,label=f'max iou (abs mean: {round(np.mean(np.abs(plt_max_iou)[plt_labels_exclude_edge]),2)})')
ax[1].scatter(plt_sizes[plt_labels_exclude_edge],plt_kdc_iou[plt_labels_exclude_edge]
              ,label=f'nearest neighbour (mean: {round(np.mean(np.array(plt_kdc_iou)[plt_labels_exclude_edge]),2)})')
ax[1].set_xlabel('grain size (log)', fontsize=20)
ax[1].set_ylabel('iou', fontsize=20)
ax[1].grid()
ax[1].legend()
ax[0].set_title(f'Excluding edge objects, no. clips: {len(all_label_exclude_edge)}, no. samples: {np.sum(plt_labels_exclude_edge)}', fontsize=20)
plt.savefig(OutDIR+f'all_dist_iou_plot_noedge.png')
plt.show()

end_script = time.time()
print('script took: ', end_script-start_script)