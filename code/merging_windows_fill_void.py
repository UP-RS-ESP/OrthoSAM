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
import sys
import json

start_script = time.time()
#load image
init_para={'OutDIR': '/DATA/vito/output/Ravi2_fnc_dw8/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'Ravi/*',
      'fid': 0,
      'crop_size': 1024,
      'resample_factor': 1/8
      }
try: 
    para_in = sys.argv[1]
    init_para.update(para_in)
    print(init_para)
except:
    print(init_para)

# Save init_para to a JSON file
with open('init_para.json', 'w') as json_file:
    json.dump(init_para, json_file, indent=4)

OutDIR=init_para.get('OutDIR')
DataDIR=init_para.get('DataDIR')
fn_img = glob.glob(DataDIR+init_para.get('DatasetName'))
fid=init_para.get('fid')

#defining clips
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')

fn_img.sort()
image = cv2.imread(fn_img[fid])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[100:-200,500:-1000]
print(fn_img[fid].split("/")[-1]+' imported')
print('Image size:', image.shape)

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

pre_para={'Downsample': {'fxy':resample_factor},
        #'Gaussian': {'kernel size':3}
        #'CLAHE':{'clip limit':2}#,
        #'Downsample': {'fxy':4},
        #'Buffering': {'crop size': crop_size}
        }
image=fnc.preprocessing_roulette(image, pre_para)

all_reseg=np.load(OutDIR+'all_reseg_mask.npy', allow_pickle=True)

#Merging windows
Aggregate_masks_noedge=[]
pred_iou_noedge=[]
for clip_window in all_reseg:
    i=clip_window['i']
    j=clip_window['j']
    for mask,score in zip(clip_window['nms mask'], clip_window['nms mask pred iou']):
        if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
            resize=np.zeros(image.shape[:-1])
            Valid_area=resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)].shape
            if Valid_area==(crop_size,crop_size):
                resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask
            else:
                resize[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*j+crop_size)]=mask[:Valid_area[0],:Valid_area[1]]
            Aggregate_masks_noedge.append(resize.astype(np.uint8))
            pred_iou_noedge.append(score)

print(f'{len(Aggregate_masks_noedge)} masks found')
#bboxes = torch.tensor([fnc.find_bounding_boxes(mask) for mask in Aggregate_masks_noedge], device=DEVICE, dtype=torch.float)
#scores = torch.tensor(pred_iou_noedge, device=DEVICE, dtype=torch.float)
#labels = torch.zeros_like(bboxes[:, 0])

#keep = batched_nms(bboxes, scores, labels, 0.35)#was 0.3
#del bboxes, scores, labels
#Aggregate_masks_noedge_nms=[Aggregate_masks_noedge[i].astype(np.uint8) for i in keep]
Aggregate_masks_noedge_nms,_=nms(Aggregate_masks_noedge,pred_iou_noedge)
#del Aggregate_masks_noedge
print(f'NMS filtered, {len(Aggregate_masks_noedge_nms)} masks remains')

#stacked_Aggregate_masks_noedge_nms=np.sum(Aggregate_masks_noedge_nms,axis=0)

stacked_Aggregate_masks_noedge_nms = np.zeros_like(Aggregate_masks_noedge_nms[0], dtype=np.uint8)

# Sum in chunks
for mask in Aggregate_masks_noedge_nms:
    stacked_Aggregate_masks_noedge_nms += mask.astype(np.uint8)

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
#del bboxes,scores,labels,keep
del Aggregate_masks_noedge, pred_iou_noedge

ar_masks=np.stack(Aggregate_masks_noedge_nms).astype(np.uint8)

#temp=np.zeros(ar_masks[0].shape)
#for i in range(ar_masks.shape[0]):
#    temp[ar_masks[i]==1]=i

#np.save(OutDIR+'Aggregate_masks_noedge_nms',temp)
del Aggregate_masks_noedge_nms

#original 5,5
dilation_size=15
#for 3000x3000, 10000    ####need to be adaptive, consider 10% of image size
min_void_size=image.shape[0]*image.shape[1]*0.001

#identify void
kernel = np.ones((dilation_size, dilation_size), np.uint8)
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
    if (region.area > min_void_size):
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
        if ((np.max([width,length])*0.9)<=crop_size):
            print(f'Downsample not necessary')
            reseg_fxy.append(1)
        else:
            factor=int((np.max([length,width])*0.9)//crop_size)+1
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

    if ((np.max([Big_width,Big_length])*0.9)<=crop_size):
        print(f'Downsample not necessary for all voids to fit in one go')
        reseg_fxy.append(1)
    else:
        factor=int((np.max([Big_length,Big_width])*0.9)//crop_size)+1
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
                resize=fnc.downsample_fnc(resize,{'target_size':image.shape[:-1][::-1]}).astype(np.uint8)
                void_pointed_reseg_resized.append(resize)
        if len(void_pointed_reseg_resized)!=0:
            ar_masks=np.vstack((ar_masks,np.stack(void_pointed_reseg_resized)))
        #list_of_mask_centroid = [fnc.get_centroid(ar_masks[i]) for i in range(ar_masks.shape[0])]
        del void_pointed_reseg_resized, void_pointed_reseg
    else:
        print('Performing third pass SAM per void')
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
                    resize=fnc.downsample_fnc(resize,{'target_size':image.shape[:-1][::-1]}).astype(np.uint8)
                    void_pointed_reseg_resized.append(resize)
            if len(void_pointed_reseg_resized)!=0:
                ar_masks=np.vstack((ar_masks,np.stack(void_pointed_reseg_resized)))
            #list_of_mask_centroid = [fnc.get_centroid(ar_masks[i]) for i in range(ar_masks.shape[0])]
            del void_pointed_reseg_resized, void_pointed_reseg


    #saving mask
    saving_void_filled=[]
    for mask in ar_masks:
        saving_void_filled.append({'mask':mask})
    np.save(OutDIR+'all_reseg_mask_void_filled',np.hstack(saving_void_filled))

    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.imshow(np.sum(ar_masks,axis=0),alpha=0.6)
    plt.axis('off')
    plt.title(f'No edge nms Stacked after\nmerging downsampled mask\nmax overlap: {np.max(np.sum(ar_masks,axis=0))}', fontsize=20)
    plt.subplot(2,2,2)
    plt.imshow(image)
    plt.imshow(np.sum(ar_masks,axis=0)>0,alpha=0.6)
    plt.axis('off')
    plt.title('Masked area after nms and\nmerging downsampled mask', fontsize=20)
    plt.subplot(2,2,3)
    plt.imshow(np.sum([ar_masks[i]*(i+1) for i in range(ar_masks.shape[0])],axis=0), cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f'Mask area after nms and\nmerging downsampled mask\n No. of mask: {ar_masks.shape[0]}', fontsize=20)
    plt.subplot(2,2,4)
    plt.imshow(image)
    plt.imshow(np.sum(ar_masks,axis=0)>1,alpha=0.6)
    plt.axis('off')
    plt.title('Overlapping area after nms and\nmerging downsampled mask', fontsize=20)
    plt.tight_layout()
    plt.savefig(OutDIR+'Merged_masks_withvoid.png')
    plt.show()
else:
    print('No void found. Minimum size threshold or dilation parameter may need to be adjusted')
end_script = time.time()
print('script took: ', end_script-start_script)
