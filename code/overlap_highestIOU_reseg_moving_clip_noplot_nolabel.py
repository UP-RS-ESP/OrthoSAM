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

start_script = time.time()
#load image
OutDIR='/DATA/vito/output/testing/'
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
    
#release ram
del list_of_cleaned_groups_reseg_masks,list_of_cleaned_groups_reseg_masks_nms,list_of_cleaned_groups_reseg_score
del list_of_cleaned_groups_reseg_score_nms, list_of_nooverlap_mask, cleaned_groups_reseg, cleaned_groups
del masks, list_of_pred_iou,list_of_stability_score,list_of_masks
del list_of_mask_centroid, ar_masks, ar_masks_flat

#np.save(OutDIR+'all_reseg_mask',np.hstack(all_reseg))

#Merging windows
Aggregate_masks_noedge=[]
pred_iou_noedge=[]
crop_size=1024
for clip_window in all_reseg:
    i=clip_window['i']
    j=clip_window['j']
    for mask,score in zip(clip_window['mask'], clip_window['mask pred iou']):
        if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
            resize=np.zeros(image.shape[:-1])
            Valid_area=resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)].shape
            if Valid_area==(crop_size,crop_size):
                resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask
            else:
                resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
            Aggregate_masks_noedge.append(resize)
            pred_iou_noedge.append(score)

bboxes = torch.tensor([fnc.find_bounding_boxes(mask) for mask in Aggregate_masks_noedge], device=DEVICE, dtype=torch.float)
scores = torch.tensor(pred_iou_noedge, device=DEVICE, dtype=torch.float)
labels = torch.zeros_like(bboxes[:, 0])

keep = batched_nms(bboxes, scores, labels, 0.35)#was 0.3
Aggregate_masks_noedge_nms=[Aggregate_masks_noedge[i] for i in keep]

stacked_Aggregate_masks_noedge_nms=np.sum(Aggregate_masks_noedge_nms,axis=0)

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

#clear RAM
del bboxes,scores,labels,keep
del Aggregate_masks_noedge, pred_iou_noedge

list_of_mask_centroid = [fnc.get_centroid(mask) for mask in Aggregate_masks_noedge_nms]
ar_masks=np.stack(Aggregate_masks_noedge_nms)
np.save(OutDIR+'Aggregate_masks_noedge_nms',ar_masks)
del Aggregate_masks_noedge_nms

#identify void
kernel = np.ones((5, 5), np.uint8)
#stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(stacked_Aggregate_masks_noedge_nms>=1, kernel)
#no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)
stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(np.sum(ar_masks,axis=0)>=1, kernel)
no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)

regions=regionprops(no_mask_area)

list_of_no_mask_area_centroid=[]
list_of_no_mask_area_mask=[]
list_of_no_mask_area_bbox=[]
for i,region in enumerate(regions):
    # take regions with large enough areas
    if (region.area > 10000):
        mask=np.array(no_mask_area==(i+1))
        if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
            y0, x0 =region.centroid
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)

            list_of_no_mask_area_centroid.append((y0,x0))
            list_of_no_mask_area_mask.append(no_mask_area==(i+1))
            list_of_no_mask_area_bbox.append([bx,by])

if len(list_of_no_mask_area_mask)>0:
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
        if (width<=1020 and length<=1020):
            print(f'Downsample not necessary')
            reseg_fxy.append(1)
        elif (width<=2040 and length<=2040):
            print('Downsampled by factor of 2 required')
            reseg_fxy.append(2)
        elif (width<=4080 and length<=4080):
            print('Downsampled by factor of 4 required')
            reseg_fxy.append(4)
        else:
            factor=np.max([length,width])//1024+1
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

    if (Big_width<=1020 and Big_length<=1020):
        print(f'Downsample not necessary for all voids to fit in one go')
        reseg_fxy.append(1)
    elif (Big_width<=2040 and Big_length<=2040):
        print('Downsampled by factor of 2 required for all voids to fit in one go')
        reseg_fxy.append(2)
    elif (Big_width<=4080 and Big_length<=4080):
        print('Downsampled by factor of 4 required for all voids to fit in one go')
        reseg_fxy.append(4)
    else:
        factor=np.max([Big_length,Big_width])//1024+1
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
        print('Performing void reseg in one go')
        fxy=1/reseg_fxy[-1]
        pre_para={'Downsample': {'fxy':fxy}}
        image_dw=fnc.preprocessing_roulette(image, pre_para)

        #define the window
        xmin=all_void_x0*fxy-512
        ymin=all_void_y0*fxy-512
        ji=xmin/1024
        ii=ymin/1024
        if ji<0:
            ji=0
        if ii<0:
            ii=0

        #prepare image
        pre_para={'Downsample': {'fxy':fxy},
                'Crop': {'crop size': crop_size, 'j':ji,'i':ii},
                'Gaussian': {'kernel size':3}
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
                resize=fnc.downsample_fnc(resize,{'fxy':reseg_fxy[-1]}).astype(int)
                void_pointed_reseg_resized.append(resize)

        ar_masks=np.vstack((ar_masks,np.stack(void_pointed_reseg_resized)))
        #list_of_mask_centroid = [fnc.get_centroid(ar_masks[i]) for i in range(ar_masks.shape[0])]
        del void_pointed_reseg_resized, void_pointed_reseg
    else:
        print('Performing void reseg per void')
        for i in range(ar_no_mask_area.shape[0]):
            void_pointed_reseg=[]
            y0, x0 =list_of_no_mask_area_centroid[i]
            fxy=1/reseg_fxy[i]
            pre_para={'Downsample': {'fxy':fxy}}
            image_dw=fnc.preprocessing_roulette(image, pre_para)
            xmin,ymin=x0*fxy-512,y0*fxy-512
            ji=xmin/1024
            ii=ymin/1024
            if ji<0:
                ji=0
            if ii<0:
                ii=0

            #prepare image
            pre_para={'Downsample': {'fxy':fxy},
                    'Crop': {'crop size': crop_size, 'j':ji,'i':ii},
                    'Gaussian': {'kernel size':3}
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
            void_pointed_reseg.append(partmasks[best_idx])

            void_pointed_reseg_resized=[]
            for mask in void_pointed_reseg:
                if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                    resize=np.zeros(image_dw.shape[:-1])
                    Valid_area=resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)].shape
                    if Valid_area==(crop_size,crop_size):
                        resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]=mask
                    else:
                        resize[int(crop_size*ii):int(crop_size*ii+crop_size),int(crop_size*ji):int(crop_size*ji+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
                    resize=fnc.downsample_fnc(resize,{'fxy':reseg_fxy[i]}).astype(int)
                    void_pointed_reseg_resized.append(resize)

            ar_masks=np.vstack((ar_masks,np.stack(void_pointed_reseg_resized)))
            #list_of_mask_centroid = [fnc.get_centroid(ar_masks[i]) for i in range(ar_masks.shape[0])]
            del void_pointed_reseg_resized, void_pointed_reseg

        

    np.save(OutDIR+'Aggregate_masks_noedge_nms_voidreseg',ar_masks)

    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.imshow(np.sum(ar_masks,axis=0),alpha=0.6)
    plt.axis('off')
    plt.title(f'No edge nms Stacked after merging downsampled mask\nmax overlap: {np.max(np.sum(ar_masks,axis=0))}', fontsize=20)
    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.imshow(np.sum(ar_masks,axis=0)>1,alpha=0.6)
    plt.axis('off')
    plt.title('Overlapping area after nms and\nmerging downsampled mask', fontsize=20)
    plt.tight_layout()
    plt.savefig(OutDIR+'Merged_masks_withvoid.png')
    plt.show()

end_script = time.time()
print('script took: ', end_script-start_script)