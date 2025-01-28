import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import clip
from PIL import Image
from tqdm import tqdm
from skimage.measure import label, regionprops
import math
from torchvision.ops.boxes import batched_nms
import pandas as pd
import psutil
import os
from multiprocessing import Pool, Manager, cpu_count

def samplot(image, mask_generator, label=None, ax=None):
    '''
    testing 
    '''
    masks = mask_generator.generate(image)
    if ax:
        ax.imshow(image)  
        show_anns_mod_ax(masks, ax, label)
        ax.axis('off')
        ax.set_title(f'Number of masks: {len(masks)}')
    else:
        fig, ax=plt.subplots(1,2,figsize=(20,20))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[1].imshow(image)
        show_anns_mod_ax(masks, ax[1], label)
        ax[1].axis('off')
        plt.title(f'Number of masks: {len(masks)}')
        plt.tight_layout()
        plt.show()
    return masks
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_anns_array(anns, ax):
    if len(anns) == 0:
        return
    ax.set_autoscale_on(False)

    img = np.ones((anns[0].shape[0], anns[0].shape[1], 4))
    img[:,:,3] = 0
    i=0
    for id in range(anns.shape[0]):
        m = anns[id].astype(int)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        i+=1

    ax.imshow(img)

def show_anns_mod(anns, label=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    i=0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        if label:
            ax.text(ann['point_coords'][0][0],ann['point_coords'][0][1], f'{i}')
        i+=1

    ax.imshow(img)

def show_anns_mod_ax(anns, ax, label=None, sort=True):
    if len(anns) == 0:
        return
    if sort:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    else:
        sorted_anns=anns
    #ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    i=0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        if label:
            ax.text(ann['point_coords'][0][0],ann['point_coords'][0][1], f'{i}')
        i+=1

    ax.imshow(img)
    
def classify_and_return_notable(image, text, labels, model):
  with torch.no_grad():
      image_features = model.encode_image(image)
      text_features = model.encode_text(text)

      logits_per_image, logits_per_text = model(image, text)
      probs = logits_per_image.softmax(dim=-1).cpu().numpy()
  most_prob = np.argmax(probs[0])
  return most_prob, probs[0][most_prob]

#Creat zero-shot classifier weights
def zeroshot_classifier(classnames, templates, DEVICE, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            if DEVICE.type=="cuda":
                texts = clip.tokenize(texts).cuda() #tokenize
            else:
                texts = clip.tokenize(texts).cpu()
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        if DEVICE.type=="cuda":
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        else:
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cpu()#.cuda()
    return zeroshot_weights

def clip_plotting(img, mask, masks, p1=0.3,p2=0.5):
    fig, ax= plt.subplots(2,2, figsize=(15,15))
    ax=ax.flatten()
    ax[0].imshow(img)
    show_anns_mod_ax(masks, ax[0])
    ax[0].set_title('RGB')
    ax[0].axis('off')
    ax[1].imshow(img)
    ax[1].imshow(mask, alpha=0.5)
    ax[1].set_title('Probability mask')
    ax[1].axis('off')
    ax[2].imshow(img)
    ax[2].imshow(mask>p1, alpha=0.5)
    ax[2].set_title(f'P>{p1}')
    ax[2].axis('off')
    ax[3].imshow(img)
    ax[3].imshow(mask>p2, alpha=0.5)
    ax[3].set_title(f'P>{p2}')
    ax[3].axis('off')
    plt.tight_layout()
    #cbar = plt.colorbar(ax[1].imshow(probability_mask, alpha=0.5), ax=ax[1], orientation='horizontal', shrink=0.5)
    #cbar.set_label('probability')

    #cbar_ax = fig.add_axes([ax[2].get_position().x0, ax[2].get_position().y0, ax[2].get_position().width, ax[2].get_position().height])
    #cbar = plt.colorbar(ax[1].imshow(probability_mask, alpha=0.5), cax=cbar_ax, orientation='vertical', shrink=0.5)
    #cbar.set_label('probability')
    plt.show()

def clip_plotting_array(img, mask, masks, p1=0.3,p2=0.5):
    fig, ax= plt.subplots(2,2, figsize=(15,15))
    ax=ax.flatten()
    ax[0].imshow(img)
    ax[0].imshow(np.sum(masks,axis=0))
    ax[0].set_title('RGB')
    ax[0].axis('off')
    ax[1].imshow(img)
    ax[1].imshow(mask, alpha=0.5)
    ax[1].set_title('Probability mask')
    ax[1].axis('off')
    ax[2].imshow(img)
    ax[2].imshow(mask>p1, alpha=0.5)
    ax[2].set_title(f'P>{p1}')
    ax[2].axis('off')
    ax[3].imshow(img)
    ax[3].imshow(mask>p2, alpha=0.5)
    ax[3].set_title(f'P>{p2}')
    ax[3].axis('off')
    plt.tight_layout()
    #cbar = plt.colorbar(ax[1].imshow(probability_mask, alpha=0.5), ax=ax[1], orientation='horizontal', shrink=0.5)
    #cbar.set_label('probability')

    #cbar_ax = fig.add_axes([ax[2].get_position().x0, ax[2].get_position().y0, ax[2].get_position().width, ax[2].get_position().height])
    #cbar = plt.colorbar(ax[1].imshow(probability_mask, alpha=0.5), cax=cbar_ax, orientation='vertical', shrink=0.5)
    #cbar.set_label('probability')
    plt.show()

def clipzeroshot(img, masks, zeroshot_weights, crop_size, target_index, model, DEVICE, preprocess, WithBackground=False, pre_mask=None):
    #load image
    pil_image = Image.fromarray(img[:crop_size,:crop_size,:])
    numpy_array = np.array(pil_image)
    if isinstance(pre_mask, np.ndarray):
            numpy_array[pre_mask==0]=0

    #output
    most_probables = []
    probabilities = []
    clip_probability_mask=np.zeros(numpy_array.shape[:-1])
    clip_top1_probability_mask=np.zeros(numpy_array.shape[:-1])

    #run prediction
    for sg in tqdm(masks, desc='Processing', unit='segment'):
        sg_mask=sg['segmentation'][:crop_size,:crop_size]

        #masking
        masked_image = np.copy(numpy_array)
        masked_image[sg_mask == 0] = 0

        #cropping
        notzero_indices = np.nonzero(masked_image)
        if len(notzero_indices[0])>0:
            x_min = max(5,np.min(notzero_indices[0]).astype(int))# with max to avoid cropping border to get negative value
            x_max = min(numpy_array.shape[0]-5, np.max(notzero_indices[0]).astype(int))# with min to avoid crossing image border
            y_min = max(5,np.min(notzero_indices[1]).astype(int))
            y_max = min(numpy_array.shape[1]-5, np.max(notzero_indices[1]).astype(int))
                
            if WithBackground:
                clip_masked_image=numpy_array[x_min-5:x_max+5, y_min-5:y_max+5,:]
            else:
                clip_masked_image=masked_image[x_min-5:x_max+5, y_min-5:y_max+5,:]

            #convert back to PIL
            if ((clip_masked_image.shape[0]!=0)&(clip_masked_image.shape[1]!=0)):
                preprocessed_pil_image = Image.fromarray(clip_masked_image)

                #predict
                image = preprocess(preprocessed_pil_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100. * image_features @ zeroshot_weights
                #output
                clip_probability_mask[sg_mask]=logits[0][target_index].cpu().detach().numpy()
                most_porbable = logits.topk(1)[1][0][0].cpu().detach().numpy()
                if int(most_porbable)<=target_index:
                    clip_top1_probability_mask[sg_mask]=logits[0][target_index].cpu().detach().numpy()#probability
            else:
                most_probables.append(np.nan)
                probabilities.append(np.nan)
    return clip_top1_probability_mask, clip_probability_mask, most_probables, probabilities

def crop_to_valid(array):
    sum_array=np.sum(array, axis=2)
    valid_mask=[(sum_array!=0)&(sum_array!=765)]
    notzero_indices = np.nonzero(valid_mask[0])
    if len(notzero_indices[0])>0:
        x_min = np.min(notzero_indices[0]).astype(int)# with max to avoid cropping border to get negative value
        x_max = np.max(notzero_indices[0]).astype(int)# with min to avoid crossing image border
        y_min = np.min(notzero_indices[1]).astype(int)
        y_max = np.max(notzero_indices[1]).astype(int)
    return array[x_min:x_max, y_min:y_max,:]
def crop_valid_no_edge(array):
    sum_array=np.sum(array, axis=2)
    valid_mask=[(sum_array!=0)&(sum_array!=765)]

    shift=1
    test=np.where(np.sum(valid_mask[0],axis=0)==(np.sum(valid_mask[0],axis=0).max()-shift))
    while len(test[0])<2:
        shift+=1
        test=np.where(np.sum(valid_mask[0],axis=0)==(np.sum(valid_mask[0],axis=0).max()-shift))
    sizes=test[0][1:]-test[0][:-1]

    while not np.any(sizes>=(np.sum(valid_mask[0],axis=0).max()-shift)):
        shift+=1
        test=np.where(np.sum(valid_mask[0],axis=0)==(np.sum(valid_mask[0],axis=0).max()-shift))
        sizes=test[0][1:]-test[0][:-1]
    
    x_idx=np.nonzero(sizes>=(np.sum(valid_mask[0],axis=0).max()-shift))[0]
    x_min,x_max=test[0][x_idx],test[0][x_idx+1]

    shift=1
    test=np.where(np.sum(valid_mask[0],axis=1)==(np.sum(valid_mask[0],axis=1).max()-shift))
    while len(test[0])<2:
        shift+=1
        test=np.where(np.sum(valid_mask[0],axis=1)==(np.sum(valid_mask[0],axis=1).max()-shift))
    sizes=test[0][1:]-test[0][:-1]

    while not np.any(sizes>=(np.sum(valid_mask[0],axis=1).max()-shift)):
        shift+=1
        test=np.where(np.sum(valid_mask[0],axis=1)==(np.sum(valid_mask[0],axis=1).max()-shift))
        sizes=test[0][1:]-test[0][:-1]
    
    y_idx=np.nonzero(sizes>=(np.sum(valid_mask[0],axis=1).max()-shift))[0]
    y_min,y_max=test[0][y_idx],test[0][y_idx+1]
    return y_min[0],y_max[0],x_min[0],x_max[0]

def clean_mask(mask):
    labels = label(mask)
    l = len(np.unique(labels))
    if l > 2:
        #get area
        regions = regionprops(labels)
        # Sort regions by area
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
        return (labels==sorted_regions[0].label).astype(np.uint8)
    else:
        return mask
    

def crop_fnc(input, para_in):
    '''
    crop size
    i
    j
    '''
    para={'crop size': 2048,
          'i':0,
          'j':0}
    para.update(para_in)
    crop_size=para.get('crop size')
    i=para.get('i')
    j=para.get('j')
    if len(input.shape)==3:
        output = input[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*(j)+crop_size),:]
    else:
        output = input[int(crop_size*i):int(crop_size*i+crop_size),int(crop_size*j):int(crop_size*(j)+crop_size)]
    #print(f'Cropped from {input.shape} to {output.shape}')
    return output
def gaussian_fnc(input, para_in):
    para={'kernel size': 3}
    para.update(para_in)
    k=para.get('kernel size')

    output = cv2.GaussianBlur(input, (k, k), k/6) 
    return output

def clahe_fnc(input, para_in):
    para={'clahe window': 50,
            'clip limit': 4
            }
    para.update(para_in)

    clahe_ws=para.get('clahe window')
    clip_limit=para.get('clip limit')

    lab_image = cv2.cvtColor(input, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    xysize=input.shape[:-1]
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(xysize[0]//clahe_ws,xysize[1]//clahe_ws))#84, 3-512
    cl_channel = clahe.apply(l_channel)

    merged_lab = cv2.merge([cl_channel, a_channel, b_channel])
    output = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return output

def Lpull_fnc(input, para_in):
    para={'thres': 60,
          'pull': 60
          }
    para.update(para_in)

    thres=para.get('thres')
    pull=para.get('pull')

    lab_image = cv2.cvtColor(input, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    cl_channel = l_channel.copy()
    cl_channel[cl_channel<thres]+=pull

    merged_lab = cv2.merge([cl_channel, a_channel, b_channel])
    output = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return output

def resample_fnc(input, para_in):
    para={'fxy':1,
          'method': None,
          'target_size': (0, 0)}
    para.update(para_in)

    fxy=para.get('fxy')
    method=para.get('method')
    target_size=para.get('target_size')
    if target_size!=(0,0):
        if method:
            if method=='area':
                output=cv2.resize(input, target_size, interpolation = cv2.INTER_AREA)
            elif method=='nearest':
                output=cv2.resize(input, target_size, interpolation = cv2.INTER_NEAREST)
        else:
            output=cv2.resize(input, target_size)
    elif fxy!=1:
        if method:
            if method=='area':
                output=cv2.resize(input, (0, 0), fx = fxy, fy = fxy, interpolation = cv2.INTER_AREA)
            elif method=='nearest':
                output=cv2.resize(input, (0, 0), fx = fxy, fy = fxy, interpolation = cv2.INTER_NEAREST)
        else:
            output=cv2.resize(input, (0, 0), fx = fxy, fy = fxy)
    else:
        print('Not resampling')
        output=input
    return output

def buffering_fnc(input,para_in):
    para={'crop size': 2048}
    para.update(para_in)

    target_size=para.get('crop size')
    if (input.shape[0]==target_size) and (input.shape[1]==target_size):
        print('Boundary not hit, buffering not required')
        return input
    else:
        if len(input.shape)<3:
            buffered=np.zeros((target_size,target_size))
            buffered[:input.shape[0],:input.shape[1]]=input
        else:
            buffered=np.zeros((target_size,target_size,3))
            for i in range(3):
                buffered[:input.shape[0],:input.shape[1],i]=input[:,:,i]
        print(f'From {input.shape} buffered to {buffered.shape}')
        return buffered


def load_roulette(input, process, para_in):
    if process=='Crop':
        output=crop_fnc(input, para_in)
    elif process=='Gaussian':
        output = gaussian_fnc(input, para_in)
    elif process=='CLAHE':
        output = clahe_fnc(input,para_in)
    elif process=='Lpull':
        output = Lpull_fnc(input,para_in)
    elif process=='Resample':
        output = resample_fnc(input,para_in)
    elif process=='Buffering':
        output = buffering_fnc(input,para_in)
    else:
        print('No process performed, returning input')
        output=input
    return output
def preprocessing_roulette(input, process_para):
    '''
    Crop:
    para={'crop size': 2048, 'i':0, 'j':0}

    Gaussian:
    para={'kernel size': 3}

    CLAHE:
    para={'clahe window': 50, 'clip limit': 4}

    Lpull:
    para={'thres': 60, 'pull': 60}

    Resample:
    para={'fxy':2, 'method': None}
    
    Buffer:
    para={'crop size': 2048}
    '''
    temp_input = input.copy()
    if len(process_para.items())>0:
        for process, para in process_para.items():
            temp_input=load_roulette(temp_input,process,para)
    else:
        print('No process performed, returning input')
    return temp_input

def get_centroid(mask):
    labels = label(mask)
    regions = regionprops(labels)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    if len(regions)>0:
        return sorted_regions[0].centroid
    else:
        return (0,0)

def make_circle(radius, array_size = 256):
    '''
    make a circle with input radius. returns a true false mask. optional arg 2 array_size, default 256
    '''
    #array = np.zeros((array_size, array_size), dtype=np.uint8)
    center = (array_size // 2, array_size // 2)
    Y, X = np.ogrid[:array_size, :array_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    #array[mask] = 1
    return mask

def circle_colouring(mask):
    '''
    colours the input circle and the dege. returns a coloured array.
    '''
    RGB = np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)
    RGB_edge = np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)
    array = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    array[mask,0] = RGB[0]
    array[mask,1] = RGB[1]
    array[mask,2] = RGB[2]
    array[~mask,0] = RGB_edge[0]
    array[~mask,1] = RGB_edge[1]
    array[~mask,2] = RGB_edge[2]
    return array, RGB, RGB_edge

def circle_colouring_specified(mask, RGB, RGB_edge):
    '''
    colours the input circle and the dege. returns a coloured array.
    '''
    array = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    array[mask,0] = RGB[0]
    array[mask,1] = RGB[1]
    array[mask,2] = RGB[2]
    array[~mask,0] = RGB_edge[0]
    array[~mask,1] = RGB_edge[1]
    array[~mask,2] = RGB_edge[2]
    return array

def add_gaussian_noise_to_circle(array, mean ,std , mask=None, edge_std=None):
    '''
    add gaussian noise to the input image. if mask is given noise will not be added to the area outside the circle. if edge_std is given, different noise will be applied to the edge. mask is required for that.
    '''
    gaussian_noise = np.random.normal(mean, std, array.shape)
    if np.any(mask):
        gaussian_noise=gaussian_noise*mask[:, :, np.newaxis]
        if edge_std:
            gaussian_noise_ed = np.random.normal(mean, edge_std, array.shape)
            gaussian_noise+=gaussian_noise_ed*~mask[:, :, np.newaxis]
    noisy_image = array.astype(float) + gaussian_noise
    #noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def iou(mask1, mask2):
    # Ensure that the input arrays are binary
    assert np.array_equal(mask1, mask1.astype(bool)), "mask1 is not binary"
    assert np.array_equal(mask2, mask2.astype(bool)), "mask2 is not binary"
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Compute IoU
    iou = intersection / union if union != 0 else 0
    
    return iou

def angular_distance(RGB_a, RGB_b):
    return math.degrees(np.arccos((RGB_a[0]*RGB_b[0]+RGB_a[1]*RGB_b[1]+RGB_a[2]*RGB_b[2])))

def euclidean_distance(color1, color2):
    R1, G1, B1 = color1
    R2, G2, B2 = color2
    distance = math.sqrt((R2 - R1)**2 + (G2 - G1)**2 + (B2 - B1)**2)
    return distance

def normalize_rgb(RGB):
    nor=np.sqrt(RGB[0]**2+RGB[1]**2+RGB[2]**2)
    nor_rgb=RGB/nor
    return nor_rgb

def mean_std_overlay(temp_image,list_of_masks,cleaned_groups,list_of_mask_centroid,k,ts,tm):
    stacked=np.stack([list_of_masks[i] for i in cleaned_groups[k]])
    mean_stacked=np.mean(stacked,axis=0)
    std_stacked=np.std(stacked,axis=0)

    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.imshow(temp_image)
    plt.imshow(mean_stacked, alpha=0.7)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('mean overlay')

    for idx in cleaned_groups[k]:
        plt.scatter(list_of_mask_centroid[idx][1],list_of_mask_centroid[idx][0])

    plt.subplot(2,2,2)
    plt.imshow(temp_image)
    plt.imshow(mean_stacked>tm, alpha=0.7)
    plt.title(f'mean>{tm} overlay')

    for idx in cleaned_groups[k]:
        plt.scatter(list_of_mask_centroid[idx][1],list_of_mask_centroid[idx][0])

    plt.subplot(2,2,3)
    plt.imshow(temp_image)
    plt.imshow(std_stacked, alpha=0.7)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('std overlay')

    for idx in cleaned_groups[k]:
        plt.scatter(list_of_mask_centroid[idx][1],list_of_mask_centroid[idx][0])


    plt.subplot(2,2,4)
    plt.imshow(temp_image)
    plt.imshow(np.logical_and(std_stacked<ts, std_stacked>0), alpha=0.7)
    plt.title(f'std<{ts} overlay')

    for idx in cleaned_groups[k]:
        plt.scatter(list_of_mask_centroid[idx][1],list_of_mask_centroid[idx][0])

    plt.tight_layout()
    plt.show()

def find_bounding_boxes(binary_mask):
    bboxes = []
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    binary_mask=torch.tensor(binary_mask, device=DEVICE, dtype=torch.float)
    labels = torch.unique(binary_mask)
    for label in labels:
        if label == 0:
            continue  # skip the background
        positions = torch.nonzero(binary_mask == label)
        if positions.numel() == 0:
            continue
        y_min, x_min = positions.min(dim=0)[0]
        y_max, x_max = positions.max(dim=0)[0]
        bboxes.append([x_min.item(), y_min.item(), x_max.item() + 1, y_max.item() + 1])
    if len(bboxes)>0:
        return bboxes[0]
    else:
        return None
def create_stats_df(masks,resample_factor=1):
    if type(masks)==np.ndarray:
        if len(masks.shape)==3:
            max_id=masks.shape[0]
            ids=np.arange(max_id)
            from_id=False
        else:
            ids=np.unique(masks)
            from_id=True
    elif type(masks)==list:
        max_id=len(masks)
        ids=np.arange(max_id)
        from_id=False
    all_stats=[]

    
    for i in ids:
        if not from_id:
            label_img = label(masks[i])
        else:
            label_img = label(masks==i)
        l = len(np.unique(label_img))

        regions = regionprops(label_img)


        if l > 2:
            #get area
            regions = regionprops(label_img)
            # Sort regions by area
            regions = sorted(regions, key=lambda x: x.area, reverse=True)
        if len(regions)>0:
            y0, x0=regions[0].centroid

            all_stats.append({'label':i,'area':regions[0].area/resample_factor
                                ,'centroid y':y0/resample_factor,'centroid x':x0/resample_factor
                                , 'major axis length': regions[0].axis_major_length/resample_factor
                                , 'minor axis length': regions[0].axis_minor_length/resample_factor
                                , 'orientation': regions[0].orientation
                                , 'perimeter': regions[0].perimeter/resample_factor
                                , 'bbox':[box/resample_factor for box in regions[0].bbox]})
    stats_df=pd.DataFrame(all_stats)
    return stats_df
        
def nms(lst_msk,lst_score):
    if len(lst_msk)>1:
        #NMS filtering
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda:0')
        else:
            DEVICE = torch.device('cpu')
        b=[find_bounding_boxes(mask) for mask in lst_msk]
        bboxes = torch.tensor([bb for bb in b if bb], device=DEVICE, dtype=torch.float)
        scores = torch.tensor([score for i,score in enumerate(lst_score) if b[i]], device=DEVICE, dtype=torch.float)
        labels = torch.zeros_like(bboxes[:, 0])

        keep = batched_nms(bboxes, scores, labels, 0.3)
        lst_msk_nms=[lst_msk[i] for i in keep]
        lst_score_nms=[lst_score[i] for i in keep]
        return lst_msk_nms,lst_score_nms
    else:
        return lst_msk,lst_score

def define_clips(x,y,resample_factor,crop_size, window_step=0.5):
    clipi=np.arange(0,(x*resample_factor)//crop_size+1,window_step)
    clipj=np.arange(0,(y*resample_factor)//crop_size+1,window_step)
    clipij=np.array(np.meshgrid(clipi, clipj)).T.reshape(-1,2)
    return clipij

def load_image(DataDIR,DSname,fid):
    fn_img = glob.glob(DataDIR+DSname)
    fn_img.sort()
    if fn_img[fid][-3:]=='npy':
        #image=(np.load(fn_img[fid])*255).astype(np.uint8)
        image=(np.load(fn_img[fid])).astype(np.uint8)
    else:
        image = cv2.imread(fn_img[fid])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(fn_img[fid].split("/")[-1]+' imported')
    return image

def set_sam(MODEL_TYPE,DataDIR):
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
    return DEVICE, CHECKPOINT_PATH

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    # Convert from bytes to MB
    return mem_info.rss / (1024 * 1024)

def get_image_patches(image, crop_size, overlap):

    H, W = image.shape[:2]
    patch_h, patch_w = crop_size, crop_size
    stride_h, stride_w = patch_h - overlap, patch_w - overlap

    patches = {}
    i, j = 0, 0
    for y in range(0, H - overlap, stride_h):
        j = 0
        for x in range(0, W - overlap, stride_w):
            patch = image[y:y + patch_h, x:x + patch_w]
            patches[(i, j)] = patch
            j += 1
        i += 1

    return patches

def untile(id_mask, patch, original_i, original_j, crop_size, overlap):

    temp_untile = np.zeros_like(id_mask, dtype=np.uint16)
    stride = crop_size - overlap

    start_y = original_i * stride
    start_x = original_j * stride
    try:
        temp_untile[start_y:start_y + patch.shape[0], start_x:start_x + patch.shape[1]] = patch
    except:
        pass
    return temp_untile

def mask_in_valid_box(mask,b, ij_idx, max_ij):
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

    area_in_box=np.sum(mask[y0:y1,x0:x1])
    total_area=np.sum(mask)
    if (total_area-area_in_box)==0:
        return True
    elif (total_area-area_in_box)<area_in_box:
        return True
    else:
        return False
    
def create_dir_ifnotexist(OutDIR):
    if not os.path.exists(OutDIR[:-1]):
        os.makedirs(OutDIR[:-1])
    if not os.path.exists(OutDIR+'chunks'):
        os.makedirs(OutDIR+'chunks')
    if not os.path.exists(OutDIR+'Merged'):
        os.makedirs(OutDIR+'Merged')
    if not os.path.exists(OutDIR+'Third'):
        os.makedirs(OutDIR+'Third')

import matplotlib.patches as patches
def plot_tiling_with_overlap(image, crop_size, overlap):
    """
    Plot the tiling of the image and highlight the overlapping areas.
    
    Parameters:
    - image: 2D or 3D NumPy array (H, W, C) or (H, W).
    - crop_size: Tuple (height, width) specifying the size of each patch.
    - overlap: The number of pixels overlapping between patches.
    """
    H, W = image.shape[:2]
    patch_h, patch_w = crop_size
    stride_h, stride_w = patch_h - overlap, patch_w - overlap

    plt.imshow(image)

    # Add patches and highlight overlaps
    for i, y in enumerate(range(0, H - overlap, stride_h)):
        for j, x in enumerate(range(0, W - overlap, stride_w)):
            # Draw the patch as a rectangle
            rect = patches.Rectangle((x, y), patch_w, patch_h, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
            # Highlight the overlapping area
            if overlap > 0:
                # Overlap along x direction
                if x + patch_w < W:
                    overlap_rect_x = patches.Rectangle((x + patch_w - overlap, y), overlap, patch_h, 
                                                       linewidth=0, facecolor='yellow', alpha=0.5)
                    plt.gca().add_patch(overlap_rect_x)
                # Overlap along y direction
                if y + patch_h < H:
                    overlap_rect_y = patches.Rectangle((x, y + patch_h - overlap), patch_w, overlap, 
                                                       linewidth=0, facecolor='yellow', alpha=0.5)
                    plt.gca().add_patch(overlap_rect_y)
                # Overlap corner (both x and y)
                if (x + patch_w < W) and (y + patch_h < H):
                    overlap_rect_xy = patches.Rectangle((x + patch_w - overlap, y + patch_h - overlap), overlap, overlap,
                                                        linewidth=0, facecolor='yellow', alpha=0.5)
                    plt.gca().add_patch(overlap_rect_xy)

    # Set the axis limits and title
    plt.xlim([0, W])
    plt.ylim([H, 0])
    plt.title("Tiling with Overlaps (highlighted in yellow)")



def get_patch_coordinates(i, j, crop_size, overlap):
    """
    Calculate the top-left corner (x0, y0) of a patch based on (i, j) indices.

    Parameters:
    - i, j: Indices of the patch in the tiling (row, column).
    - crop_size: Tuple (height, width) specifying the size of each patch.
    - overlap: The number of pixels overlapping between patches.

    Returns:
    - x0, y0: The top-left corner coordinates of the patch in the original image.
    """
    patch_h, patch_w = crop_size
    x0 = j * (patch_w - overlap)
    y0 = i * (patch_h - overlap)
    return x0, y0

def plot_grid_in_patches(image, crop_size, overlap, grid_points):
    """
    Plot the tiling of the image and add a grid of points in each patch.
    
    Parameters:
    - image: 2D or 3D NumPy array (H, W, C) or (H, W).
    - crop_size: Tuple (height, width) specifying the size of each patch.
    - overlap: The number of pixels overlapping between patches.
    - grid_points: Number of points along x and y axes in each patch.
    """
    H, W = image.shape[:2]
    patch_h, patch_w = crop_size
    stride_h, stride_w = patch_h - overlap, patch_w - overlap

    plt.imshow(image)
    
    for i, y in enumerate(range(0, H - overlap, stride_h)):
        for j, x in enumerate(range(0, W - overlap, stride_w)):
            # Calculate top-left (x0, y0) of each patch
            x0, y0 = get_patch_coordinates(i, j, crop_size, overlap)
            
            # Draw the patch as a rectangle
            rect = patches.Rectangle((x0, y0), patch_w, patch_h, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
            # Plot the grid of points inside the patch
            margin_x = patch_w * 0.1  # 10% margin from the edges
            margin_y = patch_h * 0.1  # 10% margin from the edges
            grid_spacing_x = (patch_w - 2 * margin_x) / (grid_points - 1)
            grid_spacing_y = (patch_h - 2 * margin_y) / (grid_points - 1)
            
            for gx in range(grid_points):
                for gy in range(grid_points):
                    point_x = x0 + margin_x + gx * grid_spacing_x
                    point_y = y0 + margin_y + gy * grid_spacing_y
                    plt.plot(point_x, point_y, 'ro', markersize=2)  # Plot blue dots
            
            # Highlight the overlapping area (optional)
            if overlap > 0:
                # Overlap along x direction
                if x0 + patch_w < W:
                    overlap_rect_x = patches.Rectangle((x0 + patch_w - overlap, y0), overlap, patch_h, 
                                                       linewidth=0, facecolor='yellow', alpha=0.5)
                    plt.gca().add_patch(overlap_rect_x)
                # Overlap along y direction
                if y0 + patch_h < H:
                    overlap_rect_y = patches.Rectangle((x0, y0 + patch_h - overlap), patch_w, overlap, 
                                                       linewidth=0, facecolor='yellow', alpha=0.5)
                    plt.gca().add_patch(overlap_rect_y)
                # Overlap corner (both x and y)
                if (x0 + patch_w < W) and (y0 + patch_h < H):
                    overlap_rect_xy = patches.Rectangle((x0 + patch_w - overlap, y0 + patch_h - overlap), overlap, overlap,
                                                        linewidth=0, facecolor='yellow', alpha=0.5)
                    plt.gca().add_patch(overlap_rect_xy)

    # Set the axis limits and title
    plt.xlim([0, W])
    plt.ylim([H, 0])
    plt.title(f"Tiling with {grid_points}x{grid_points} Point Grid in Each Patch")

def area_radi(mask, min_pixel, min_radi):
    labels = label(mask)
    regions = regionprops(labels)
    regions = sorted(regions, key=lambda x: x.area, reverse=True)
    if (regions[0].area>min_pixel and regions[0].axis_minor_length>min_radi):
        return True
    else:
        return False
    

def compute_iou(args):
    i, centroid, mask, ids, mask_ious, seg_masks_rs, seg_id = args
    hit_id = int(mask[int(centroid[0]), int(centroid[1])])
    current_iou = mask_ious[ids == hit_id]
    iou_s = iou(seg_masks_rs == seg_id, mask == hit_id)
    return i, hit_id, iou_s, current_iou

def update_mask_ious_parallel(centroids, mask, ids, seg_masks_rs, seg_ids):
    '''
    multiprocessing mask iou
    '''
    mask_ious = np.zeros_like(ids).astype(np.float64)

    # Prepare data for parallel processing
    data = [(i, centroids[i], mask, ids, mask_ious, seg_masks_rs, seg_ids[i]) for i in range(len(centroids))]

    # Use multiprocessing Pool
    with Pool(cpu_count()) as pool:
        results = pool.map(compute_iou, data)

    # Update mask_ious with results
    for i, hit_id, iou_s, current_iou in results:
        if iou_s > current_iou:
            mask_ious[ids == hit_id] = iou_s

    return mask_ious

def compute_iou_shared(args):
    i, centroid, mask, ids, mask_ious_dict, seg_masks_rs, seg_id = args
    hit_id = int(mask[int(centroid[0]), int(centroid[1])])
    current_iou = mask_ious_dict.get(hit_id, 0.0) 
    
    iou_s = iou(seg_masks_rs == seg_id, mask == hit_id)
    if np.sum(seg_masks_rs == seg_id)<np.sum(mask == hit_id):#negative for oversegmentation
        iou_s=-iou_s
    
    # Update the dictionary only if IoU improves
    if np.abs(iou_s) > np.abs(current_iou):
        mask_ious_dict[hit_id] = iou_s

def update_mask_ious_shared(centroids, mask, ids, seg_masks_rs, seg_ids):
    '''
    Multiprocessing mask IoU computation with shared memory
    '''
    with Manager() as manager:
        # Create a shared dictionary for mask_ious
        mask_ious_dict = manager.dict()

        # Prepare data for parallel processing
        data = [
            (i, centroids[i], mask, ids, mask_ious_dict, seg_masks_rs, seg_ids[i]) 
            for i in range(len(centroids))
        ]

        # Use multiprocessing Pool
        with Pool(cpu_count()) as pool:
            pool.map(compute_iou_shared, data)

        # Convert the dictionary back to a NumPy array
        mask_ious = np.zeros_like(ids).astype(np.float64)
        for hit_id, iou_val in mask_ious_dict.items():
            mask_ious[ids == hit_id] = iou_val

        return mask_ious