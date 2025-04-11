import torch
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator_mod as SamAutomaticMaskGenerator
import numpy as np
import torch
import functions as fnc
import gc
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
#testing variation in fg bg noise without changing the color. no color contrast in this case

#prep
OutDIR='/DATA/vito/output/fg_bg_noise_withsignedIOU/'
DataDIR='/DATA/vito/data/'
MODEL_TYPE = "vit_h"
if not os.path.exists(OutDIR[:-1]):
    os.makedirs(OutDIR[:-1])

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

#load record or initiate new
try:
    test_outputs=np.load(OutDIR+'collect_test.npy', allow_pickle=True).tolist()
    print('record loaded')
except:
    test_outputs=[]
    print('No record found, created new list')

#make test color pair
#color_spacing=np.arange(0,1.1,0.05)
#color_pair=np.array(np.meshgrid(color_spacing, color_spacing)).T.reshape(-1,2)

radi=[4,6,8,10]
stds_all=np.arange(0,100,5)
std_edges=np.arange(0,100,5)
for i in range(5):
    stds=stds_all[i*4:i*4+4]
    #run test
    for run in range(len(std_edges)):
        inner=True#
        #inner=False
        var=True#
        #var=False

        #create image and noise
        all_col=[]
        all_col_mask=[]
        label=1
        for r in radi:
            temp_row=[]
            temp_row_mask=[]
            for std in stds:
                mask=fnc.make_circle(r)
                rgb_sphere=np.zeros((256,256,3))#fnc.circle_colouring(mask)
                if inner:
                    if var:
                        edge=std_edges[run]
                        noisy_image=fnc.add_guassian_noise_to_circle(rgb_sphere,0,std,mask,edge)
                    else:
                        noisy_image=fnc.add_guassian_noise_to_circle(rgb_sphere,0,std,mask)
                else:
                    noisy_image=fnc.add_guassian_noise_to_circle(rgb_sphere,0,std)
                temp_row.append(noisy_image)
                temp_row_mask.append((mask).astype(int)*label)
                label+=1
            all_col.append(np.vstack(temp_row))
            all_col_mask.append(np.vstack(temp_row_mask))
        noise_layer=np.hstack(all_col)
        all_mask=np.hstack(all_col_mask)

        #All_nearest_iou=[]
        All_max_iou=[]
        
        #for channel in range(3):
        RGB=[128,128,128]
        RGB_edge=[128,128,128]
        #RGB[channel]=int(color_pair[run][0]*255)
        #RGB_edge[channel]=int(color_pair[run][1]*255)
        image=fnc.circle_colouring_specified(all_mask!=0, RGB, RGB_edge)

        #clipping the noise by the half of the smaller std
        #clipping=np.min([std,edge])
        #noise_layer=np.clip(noise_layer,-clipping/2, np.max(noise_layer))
        
        image=image+noise_layer
        image=np.clip(image, 0, 255).astype(np.uint8)

        #calculate color contrast
        #RGB_nor=fnc.normalize_rgb(RGB)
        #RGB_edge_nor=fnc.normalize_rgb(RGB_edge)

        #calculate distance
        #ad=fnc.angular_distance(RGB_nor,RGB_edge_nor)
        #ed=fnc.euclidean_distance(RGB,RGB_edge)

        #apply guassian filter
        #xl=[1,3,5,7,9,11,13,15,17,19,21]
        #for k in xl:
        temp_image=fnc.preprocessing_roulette(image, 
                                            {#'Crop': {'crop size': 1024, 'j':0},
                                                #'Gaussian': {'kernel size':k},
                                                #'CLAHE':{'clip limit':3}#,
                                                # #'Downsample': {'fxy':4}
                                                })
        gc.collect()
        torch.cuda.empty_cache()

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            pred_iou_thresh=0,
            stability_score_thresh=0,#iou by varying cutoff in binary conversion
            box_nms_thresh=0.3,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks
            crop_n_layers=0,#cut into 2**n crops
            crop_nms_thresh=0,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks between crops
            crop_n_points_downscale_factor=1,
            crop_overlap_ratio=0,
            #min_mask_region_area=2000,
        )
        with torch.no_grad():
            masks = mask_generator.generate(temp_image)
        #logit_scale=1000

        #prep selection
        seg_labels=(np.unique(all_mask)[1:])

        seg_size=[]
        for seg in seg_labels:
            seg_size.append(np.sum(all_mask==seg))

        seg_size_sort_idx=np.argsort(seg_size)

        list_of_label=[(all_mask==seg_labels[index]).astype(np.uint8) for index in seg_size_sort_idx]
        ar_label=np.stack(list_of_label)

        list_of_masks = [fnc.clean_mask(mask['segmentation'].astype(np.uint8)) for mask in masks]#remove small disconnected parts
        ar_masks=np.stack(list_of_masks)

        ar_all=np.stack(list_of_label+list_of_masks)
        list_of_centroid = [fnc.get_centroid(ar_all[i]) for i in range(len(ar_all))]
        ar_all.shape

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

        list_overlap = list(set_overlap)

        #centroid based selection
        #tree = KDTree(list_of_centroid)              
        #k=10
        #_, ind = tree.query(list_of_centroid[:len(list_of_label)], k=k)

        # get nearest centroid iou
        #kdc_iou=[]
        #nearest_mask_pick=[]
        #for match in ind:
            #check if nearest neighbor is not a label
        #    j=1
        #    while match[j]<len(list_of_label):
        #        j+=1
        #    if j<k:
        #        kdc_iou.append(fnc.iou(ar_all[match[0]],ar_all[match[j]]))
        #        nearest_mask_pick.append(match[j])
        #    else:
        #        kdc_iou.append(-0.1)

            
        # get all average and max iou
        overlap_by_label=[]
        max_iou=[]
        #max_mask_pick=[]
        for i in range(len(list_of_label)):
            all_pairs=[tup for tup in list_overlap if i in tup]
            if len(all_pairs)>0:
                overlap_with_i=np.unique(np.hstack(all_pairs))[1:]
                overlap_by_label.append(overlap_with_i)
                agg_iou=[]
                for j in overlap_with_i:
                    agg_iou.append(fnc.iou(ar_label[i],ar_masks[j-len(list_of_label)]))
                best_msk_idx=np.argmax(agg_iou)
                if (np.sum(ar_label[i])>np.sum(ar_masks[overlap_with_i[best_msk_idx]-len(list_of_label)])):
                    max_iou.append(-np.max(agg_iou))#assign negative iou when oversegmented
                else:
                    max_iou.append(np.max(agg_iou))
                #max_mask_pick.append(overlap_with_i[np.argmax(agg_iou)])
            else:
                overlap_by_label.append(np.array([]))
                max_iou.append(-0.1)
        #All_nearest_iou.append(kdc_iou)
        All_max_iou.append(max_iou)


        #All_nearest_iou=np.array(All_nearest_iou)
        All_max_iou=np.array(All_max_iou)
        output={#'RGB':color_pair[run][0], 'RGB_edge':color_pair[run][1], 
                #'angular_distance': ad, 'euclidean_distance':ed, 
                'Center Noise':std, 
                'Edge Noise':edge,
                'All_max_iou': All_max_iou#, 'All_nearest_iou':All_nearest_iou
                }
        test_outputs.append(output)
        #np.save(OutDIR+f'test{run}.npy',output)
        np.save(OutDIR+f'collect_test_small.npy',test_outputs)


max_iou=[out['All_max_iou'] for out in test_outputs]

ticks=range(len(std_edges))
labels=(std_edges).astype(int)

plt.figure(figsize=(15, 15))
for i in range(4):#loop through radi
    #plt.figure(figsize=(15, 30))
    #for j in range(4):#loop through noise level
    col=i*4
    plt.subplot(2, 2, i+1)
    #plt.subplot(4, 4, i*4+j+1)
    plt.title(f'Circle radius: {radi[i]}')
    plt.imshow(np.vstack(
        np.hstack(
            [np.array([arg[:,col:col+4] for arg in max_iou[:20]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[20:40]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[40:60]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[60:80]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[80:]])[:,0,:]]
            )
        )
        , origin='lower'#, aspect='auto')
                , vmax=0.9,vmin=-0.9, aspect='auto')
    plt.yticks(ticks, labels)
    plt.ylabel('background noise (std)', fontsize=20)
    #plt.xticks(xticks, ['R','G','B'])
    plt.xticks(ticks, labels)
    plt.xlabel('foreground noise (std)', fontsize=20)
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f'max iou varying noise level', fontsize=26)
plt.savefig(OutDIR+f'max_iou_small.png') 
plt.show()



##run big circles
test_outputs=[]

radi=[6,32,64,128]
stds_all=np.arange(0,100,5)
std_edges=np.arange(0,100,5)
for i in range(5):
    stds=stds_all[i*4:i*4+4]
    #run test
    for run in range(len(std_edges)):
        inner=True#
        #inner=False
        var=True#
        #var=False

        #create image and noise
        all_col=[]
        all_col_mask=[]
        label=1
        for r in radi:
            temp_row=[]
            temp_row_mask=[]
            for std in stds:
                mask=fnc.make_circle(r)
                rgb_sphere=np.zeros((256,256,3))#fnc.circle_colouring(mask)
                if inner:
                    if var:
                        edge=std_edges[run]
                        noisy_image=fnc.add_guassian_noise_to_circle(rgb_sphere,0,std,mask,edge)
                    else:
                        noisy_image=fnc.add_guassian_noise_to_circle(rgb_sphere,0,std,mask)
                else:
                    noisy_image=fnc.add_guassian_noise_to_circle(rgb_sphere,0,std)
                temp_row.append(noisy_image)
                temp_row_mask.append((mask).astype(int)*label)
                label+=1
            all_col.append(np.vstack(temp_row))
            all_col_mask.append(np.vstack(temp_row_mask))
        noise_layer=np.hstack(all_col)
        all_mask=np.hstack(all_col_mask)

        All_max_iou=[]
        
        RGB=[128,128,128]
        RGB_edge=[128,128,128]
        image=fnc.circle_colouring_specified(all_mask!=0, RGB, RGB_edge)

        
        image=image+noise_layer
        image=np.clip(image, 0, 255).astype(np.uint8)

        temp_image=fnc.preprocessing_roulette(image, 
                                            {#'Crop': {'crop size': 1024, 'j':0},
                                                #'Gaussian': {'kernel size':k},
                                                #'CLAHE':{'clip limit':3}#,
                                                # #'Downsample': {'fxy':4}
                                                })
        gc.collect()
        torch.cuda.empty_cache()

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            pred_iou_thresh=0,
            stability_score_thresh=0,#iou by varying cutoff in binary conversion
            box_nms_thresh=0.3,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks
            crop_n_layers=0,#cut into 2**n crops
            crop_nms_thresh=0,#The box IoU cutoff used by non-maximal suppression to filter duplicate masks between crops
            crop_n_points_downscale_factor=1,
            crop_overlap_ratio=0,
            #min_mask_region_area=2000,
        )
        with torch.no_grad():
            masks = mask_generator.generate(temp_image)
 

        #prep selection
        seg_labels=(np.unique(all_mask)[1:])

        seg_size=[]
        for seg in seg_labels:
            seg_size.append(np.sum(all_mask==seg))

        seg_size_sort_idx=np.argsort(seg_size)

        list_of_label=[(all_mask==seg_labels[index]).astype(np.uint8) for index in seg_size_sort_idx]
        ar_label=np.stack(list_of_label)

        list_of_masks = [fnc.clean_mask(mask['segmentation'].astype(np.uint8)) for mask in masks]#remove small disconnected parts
        ar_masks=np.stack(list_of_masks)

        ar_all=np.stack(list_of_label+list_of_masks)
        list_of_centroid = [fnc.get_centroid(ar_all[i]) for i in range(len(ar_all))]
        ar_all.shape

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

        list_overlap = list(set_overlap)

        # get all average and max iou
        overlap_by_label=[]
        max_iou=[]
        #max_mask_pick=[]
        for i in range(len(list_of_label)):
            all_pairs=[tup for tup in list_overlap if i in tup]
            if len(all_pairs)>0:
                overlap_with_i=np.unique(np.hstack(all_pairs))[1:]
                overlap_by_label.append(overlap_with_i)
                agg_iou=[]
                for j in overlap_with_i:
                    agg_iou.append(fnc.iou(ar_label[i],ar_masks[j-len(list_of_label)]))
                best_msk_idx=np.argmax(agg_iou)
                if (np.sum(ar_label[i])>np.sum(ar_masks[overlap_with_i[best_msk_idx]-len(list_of_label)])):
                    max_iou.append(-np.max(agg_iou))#assign negative iou when oversegmented
                else:
                    max_iou.append(np.max(agg_iou))
                #max_mask_pick.append(overlap_with_i[np.argmax(agg_iou)])
            else:
                overlap_by_label.append(np.array([]))
                max_iou.append(-0.1)
        #All_nearest_iou.append(kdc_iou)
        All_max_iou.append(max_iou)


        #All_nearest_iou=np.array(All_nearest_iou)
        All_max_iou=np.array(All_max_iou)
        output={#'RGB':color_pair[run][0], 'RGB_edge':color_pair[run][1], 
                #'angular_distance': ad, 'euclidean_distance':ed, 
                'Center Noise':std, 
                'Edge Noise':edge,
                'All_max_iou': All_max_iou#, 'All_nearest_iou':All_nearest_iou
                }
        test_outputs.append(output)
        #np.save(OutDIR+f'test{run}.npy',output)
        np.save(OutDIR+f'collect_test_big.npy',test_outputs)


max_iou=[out['All_max_iou'] for out in test_outputs]

ticks=range(len(std_edges))
labels=(std_edges).astype(int)

plt.figure(figsize=(15, 15))
for i in range(4):#loop through radi
    col=i*4
    plt.subplot(2, 2, i+1)
    #plt.subplot(4, 4, i*4+j+1)
    plt.title(f'Circle radius: {radi[i]}')
    plt.imshow(np.vstack(
        np.hstack(
            [np.array([arg[:,col:col+4] for arg in max_iou[:20]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[20:40]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[40:60]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[60:80]])[:,0,:],
             np.array([arg[:,col:col+4] for arg in max_iou[80:]])[:,0,:]]
            )
        )
        , origin='lower'#, aspect='auto')
                , vmax=0.9,vmin=-0.9, aspect='auto')
    plt.yticks(ticks, labels)
    plt.ylabel('background noise (std)', fontsize=20)
    #plt.xticks(xticks, ['R','G','B'])
    plt.xticks(ticks, labels)
    plt.xlabel('foreground noise (std)', fontsize=20)
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f'max iou varying noise level', fontsize=26)
plt.savefig(OutDIR+f'max_iou_big.png') 
plt.show()