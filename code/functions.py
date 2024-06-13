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
    output = input[crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1),:]
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

def downsample_fnc(input, para_in):
    para={'fxy':2,
          'inter_area': None}
    para.update(para_in)

    fxy=para.get('fxy')
    if para.get('inter_area'):
        output=cv2.resize(input, (0, 0), fx = fxy, fy = fxy, interpolation = cv2.INTER_AREA)
    else:
        output=cv2.resize(input, (0, 0), fx = fxy, fy = fxy)
    return output


def load_roulette(input, process, para_in):
    if process=='Crop':
        output=crop_fnc(input, para_in)
    elif process=='Gaussian':
        output = gaussian_fnc(input, para_in)
    elif process=='CLAHE':
        output = clahe_fnc(input,para_in)
    elif process=='Lpull':
        output = Lpull_fnc(input,para_in)
    elif process=='Downsample':
        output = downsample_fnc(input,para_in)
    else:
        print('No process performed, returning input')
        output=input
    return output
def preprocessing_roulette(input, process_para):
    '''
    Crop:
    para={'crop size': 2048,
          'i':0,
          'j':0}

    Gaussian:
    para={'kernel size': 3}

    CLAHE:
    para={'clahe window': 50,
            'clip limit': 4}

    Lpull:
    para={'thres': 60,
          'pull': 60}

    Downsample:
    para={'fxy':2,
          'inter_area': None}
    '''
    temp_input = input.copy()
    for process, para in process_para.items():
        temp_input=load_roulette(temp_input,process,para)
    return temp_input