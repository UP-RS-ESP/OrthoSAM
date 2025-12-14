import glob
import numpy as np
from tifffile import imread, imwrite, TiffFile
import cv2
import torch
from skimage.measure import label, regionprops,regionprops_table
import os
import json
from segment_anything import sam_model_registry
from torchvision.ops.boxes import batched_nms
import psutil
from tqdm import tqdm
import sys
from multiprocessing import Pool, Manager, cpu_count
import requests
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial import cKDTree as kdtree

def notify(text):
    '''
    Send a Discord alert with the given text message.

    Arguments:
    - text (str): The message to be sent to the Discord webhook.
    
    Returns:
    - None: Sends a message to the Discord webhook specified in DWH.txt.
    '''
    def send_discord_alert(webhook_url, message):
        data = {"content": message}
        try:
            response = requests.post(webhook_url, json=data)
            if response.status_code == 204:
                print("Message sent successfully to Discord!")
            else:
                print(f"Failed to send message: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # .py script
    except NameError:
        base_dir = os.path.dirname(os.path.abspath("__file__"))  # Jupyter fallback

    dwh_path = os.path.join(base_dir, 'DWH.txt')
    with open(dwh_path, 'r') as file:
        DWH = file.readline().strip()

    send_discord_alert(DWH, text)

def load_image(DataDIR,DSname,fid):
    '''
    Load image from specified directory and dataset name using file identifier.

    Arguments:
    - DataDIR (str): Directory where the dataset is located.
    - DSname (str): Name of the dataset.
    - fid (int or str): File identifier for the image to load.

    Returns:
    - image (numpy.ndarray): The loaded image.
    '''
    if isinstance(fid, int):
        fn_img = glob.glob(os.path.join(DataDIR,DSname,'*'))
        fn_img.sort()
        #print(fn_img)
        if fn_img[fid][-3:]=='npy':
            #image=(np.load(fn_img[fid])*255).astype(np.uint8)
            image=(np.load(fn_img[fid])).astype(np.uint8)
        elif fn_img[fid][-3:]=='tif':
            image = imread(fn_img[fid])[:,:,:3]
        else:
            image = cv2.imread(fn_img[fid])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(fn_img[fid]+' imported')
    elif isinstance(fid, str):
        fn=os.path.join(DataDIR,DSname,fid)
        if fn[-3:]=='npy':
            image=(np.load(fn)).astype(np.uint8)
        elif fn[-3:]=='tif':
            image = imread(fn)[:,:,:3]
        else:
            image = cv2.imread(fn)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(fn+' imported')
    return image


def crop_fnc(input, para_in):
    '''
    Crop the input image based on specified parameters.
    Arguments:
    - input (numpy.ndarray): The input image to be cropped.
    - para_in (dict): Dictionary of parameters for cropping.
        - crop size (int): Size of the crop.
        - i (int): Row index of the crop.
        - j (int): Column index of the crop.

    Returns:
    - output (numpy.ndarray): The cropped image.
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
    """
    Apply Gaussian blur to the input image.

    Arguments:
    - input (numpy.ndarray): The input image.
    - para_in (dict): Dictionary of parameters for Gaussian blur.

    Returns:
    - output (numpy.ndarray): The blurred image.
    """
    para={'kernel size': 3}
    para.update(para_in)
    k=para.get('kernel size')

    output = cv2.GaussianBlur(input, (k, k), k/6) 
    return output

def clahe_fnc(input, para_in):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.
    Arguments:
    - input (numpy.ndarray): The input image.
    - para_in (dict): Dictionary of parameters for CLAHE.
        - clahe window (int): Size of the CLAHE window.
        - clip limit (float): Clip limit for CLAHE.

    Returns:
    - output (numpy.ndarray): The CLAHE-adjusted image.
    """
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
    """
    Apply Lpull to the input image.

    Arguments:
    - input (numpy.ndarray): The input image.
    - para_in (dict): Dictionary of parameters for Lpull.
        - thres (int): Threshold value for L channel.
        - pull (int): Pull value to adjust L channel.

    Returns:
    - output (numpy.ndarray): The Lpull-adjusted image.
    """
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
    '''
    Resample the input image based on specified parameters.

    Arguments:
    - input (numpy.ndarray): The input image to be resampled.
    - para_in (dict): Dictionary of parameters for resampling.
        - fxy (float): Scaling factor.
        - method (str): Interpolation method.
        - target_size (tuple): Target size of the image.

    Returns:
    - output (numpy.ndarray): The resampled image.
    '''
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
    '''
    Buffer the input image to a target size.
    Arguments:
    - input (numpy.ndarray): The input image to be buffered.
    - para_in (dict): Dictionary of parameters for buffering.
        - crop size (int): Target size of the image.

    Returns:
    - output (numpy.ndarray): The buffered image.
    '''
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
    '''
    Load and apply a specific preprocessing function based on the process name.
    
    Arguments:
    - input (numpy.ndarray): The input image.
    - process (str): The name of the preprocessing function to apply.
    - para_in (dict): Dictionary of parameters for the preprocessing function.

    Returns:
    - output (numpy.ndarray): The preprocessed image.
    '''
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
    Apply a series of preprocessing functions to the input image based on the provided parameters.
    
    Arguments:
    - input (numpy.ndarray): The input image.
    - process_para (dict): Dictionary of preprocessing parameters.
        Options:
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
    
    Returns:
    - output (numpy.ndarray): The preprocessed image.
    '''
    temp_input = input.copy()
    if len(process_para.items())>0:
        for process, para in process_para.items():
            temp_input=load_roulette(temp_input,process,para)
    else:
        print('No process performed, returning input')
    return temp_input

def get_image_patches(image, crop_size, overlap):
    '''
    Patch image with specified crop size and overlap.
    
    Arguments:
    - image (numpy.ndarray): The input image.
    - crop_size (int): Size of the patches.
    - overlap (int): Overlap between patches.

    Returns:
    - patches (dict): Dictionary of image patches indexed by (i, j).
    '''

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

def load_config(filename='config.json'):
    '''
    Load configuration parameters from a JSON file.

    Arguments:
    - filename (str): Name of the JSON configuration file.

    Returns:
    - config (dict): Dictionary of configuration parameters.
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, filename)

    with open(config_path, 'r') as f:
        return json.load(f)
    
def set_sam(MODEL_TYPE,CheckpointDIR):
    '''
    Set up the SAM model with specified type and checkpoint directory.
    Arguments:
    - MODEL_TYPE (str): Type of the SAM model.
    - CheckpointDIR (str): Directory containing the model checkpoint.

    Returns:
    - sam (torch.nn.Module): The loaded SAM model.
    '''
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        print('Currently running on GPU\nModel '+MODEL_TYPE)
    else:
        DEVICE = torch.device('cpu')
        print('Currently running on CPU\nModel '+MODEL_TYPE)

    if MODEL_TYPE.lower() == 'vit_h':
        CHECKPOINT_PATH = os.path.join(CheckpointDIR,'sam_vit_h_4b8939.pth')
    elif MODEL_TYPE.lower() == 'vit_l':
        CHECKPOINT_PATH = os.path.join(CheckpointDIR,'sam_vit_l_0b3195.pth')
    else:
        CHECKPOINT_PATH = os.path.join(CheckpointDIR,'sam_vit_b_01ec64.pth')
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    return sam

def clean_mask(mask):
    '''
    Clean the mask by retaining only the largest connected component.
    Arguments:
    - mask (numpy.ndarray): The input binary mask.

    Returns:
    - cleaned_mask (numpy.ndarray): The cleaned binary mask with only the largest component.
    '''

    labels = label(mask)
    l = len(np.unique(labels))
    if l > 2:
        #get area
        #regions = regionprops(labels)
        ids, areas = np.unique(labels, return_counts=True)
        areas = areas[ids > 0]
        ids = ids[ids > 0]
        # Sort regions by area
        #sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
        return (labels==ids[np.argmax(areas)]).astype(np.uint8)
    else:
        return mask
    
def area_radi(mask, min_pixel, min_radi):
    '''
    Calculate if the mask has area and radius above specified minimum.
    Arguments:
    - mask (numpy.ndarray): The input binary mask.
    - min_pixel (int): Minimum area threshold.
    - min_radi (float): Minimum radius threshold.

    Returns:
    - bool: True if both area and radius exceed minimums, False otherwise.
    '''
    labels = label(mask)
    try:
        regions = regionprops(labels)
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        if (regions[0].area>min_pixel and regions[0].axis_minor_length>min_radi):
            return True
        else:
            return False
    except:
        return False

def find_bounding_boxes(binary_mask):
    '''
    Find bounding boxes for each connected component in the binary mask.
    Arguments:
    - binary_mask (numpy.ndarray): The input binary mask.

    Returns:
    - bboxes (list): List of bounding boxes for each connected component.
    '''
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
    
def nms(lst_msk,lst_score):
    '''
    Apply Non-Maximum Suppression (NMS) to a list of masks and their corresponding scores.
    Arguments:
    - lst_msk (list): List of binary masks.
    - lst_score (list): List of scores corresponding to each mask.
    
    Returns:
    - tuple: A tuple containing two lists - the filtered masks and their corresponding scores after NMS.
    '''
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
    
def get_memory_usage():
    '''
    Get the current memory usage of the process in MB.

    Returns:
    - (float): Memory usage in megabytes (MB).
    '''
    process = psutil.Process()
    mem_info = process.memory_info()
    # Convert from bytes to MB
    return mem_info.rss / (1024 * 1024)

def untile(id_mask, patch, original_i, original_j, crop_size, overlap):
    '''
    Untile a patch back into the full image mask at the specified position.
    Arguments:
    - id_mask (numpy.ndarray): The full image mask to update.
    - patch (numpy.ndarray): The patch to be untiled.
    - original_i (int): Row index of the patch in the tiled image.
    - original_j (int): Column index of the patch in the tiled image.
    - crop_size (int): Size of the patches.
    - overlap (int): Overlap between patches.

    Returns:
    - temp_untile (numpy.ndarray): The updated full image mask with the patch untiled.
    '''

    temp_untile = np.zeros_like(id_mask, dtype=np.uint16)
    stride = crop_size - overlap

    start_y = original_i * stride
    start_x = original_j * stride
    try:
        temp_untile[start_y:start_y + patch.shape[0], start_x:start_x + patch.shape[1]] = patch
    except:
        pass
    return temp_untile

def clean_and_overwrite(mask):
    '''
    Clean the mask by removing small disconnected components within each labeled region and relabeling.
    Arguments:
    - mask (numpy.ndarray): The input labeled mask.

    Returns:
    - cleaned_mask (numpy.ndarray): The cleaned labeled mask.
    '''
    labeled = label(mask)
    id_to_remove = []
    for i in tqdm(np.unique(mask)[1:], file=sys.stdout):
        if len(np.unique(labeled[mask==i]))>1:
            labels, counts = np.unique(labeled[mask==i], return_counts=True)   
            max_label = labels[np.argmax(counts)]
            id_to_remove.append(labels[labels != max_label])
    if len(id_to_remove)>0:
        for i in np.hstack(id_to_remove):
            mask[labeled==i]=0
    return label(mask)

def create_dir_ifnotexist(OutDIR):
    '''
    Create output directories if they do not exist.
    Arguments:
    - OutDIR (str): The main output directory.

    Returns:
    - None: Creates directories if they do not exist.
    '''
    if not os.path.exists(OutDIR):
        os.makedirs(OutDIR)
    if not os.path.exists(os.path.join(OutDIR,'chunks')):
        os.makedirs(os.path.join(OutDIR,'chunks'))
    if not os.path.exists(os.path.join(OutDIR,'Merged')):
        os.makedirs(os.path.join(OutDIR,'Merged'))

def prompt_fid(para):
    '''
    Prompt the user to select a file identifier (fid) from the available images in the dataset.
    Arguments:
    - para (dict): Dictionary of parameters including 'DataDIR' and 'DatasetName'.

    Returns:
    - para (dict): Updated dictionary of parameters with selected 'fid'.
    '''
    fn_img = glob.glob(os.path.join(para.get('DataDIR'),para.get('DatasetName'),'*'))
    fn_img.sort()
    for i,fn in enumerate(fn_img):
        print(i, ': ', fn)
    print('--------------')
    while True:
        try:
            user_input = int(input("Please select an image: "))
            print(f"{fn_img[user_input]} selected")
            para.update({'fid':user_input})
            break  # Exit the loop if the input is valid
        except ValueError:
            print("Requires an index. Please try again.")
    return para

def setup(master_para, para_list, pre_para_list=None):
    '''
    Setup the output directory and save parameters to JSON files.
    Arguments:
    - master_para (dict): Dictionary of master parameters.
    - para_list (list): List of parameter dictionaries for each layer.
    - pre_para_list (list, optional): List of preprocessing parameter dictionaries for each layer.

    Returns:
    - lst (list): List of combined parameter dictionaries for each layer.
    '''
    master_para['1st_resample_factor'] = master_para['resample_factor']
    config = load_config()
    master_para={**config,**master_para}
    master_para['OutDIR'] = os.path.join(master_para.get('MainOutDIR'), master_para.get('OutDIR'))
    if not os.path.exists(os.path.join(master_para.get('DataDIR'),master_para.get('DatasetName'))):
        print('Input directory does not exist. Exiting script.')
        sys.exit()

    # create dir if output dir not exist
    OutDIR=master_para.get('OutDIR')
    create_dir_ifnotexist(OutDIR)
    if master_para.get('fid')==None:
        master_para=prompt_fid(master_para)


    # Save para to a JSON file
    para_list.insert(0, {})
    lst = [dict(master_para, **para) for para in para_list]
    with open(os.path.join(OutDIR,'para.json'), 'w') as json_file:
        json.dump(lst, json_file, indent=4)
    if pre_para_list:
        with open(os.path.join(OutDIR,'pre_para.json'), 'w') as json_file:
            json.dump(pre_para_list, json_file, indent=4)
    return lst

def get_patch_at(image, i, j, crop_size, overlap):
    '''
    Get a specific patch from the image based on the provided indices.
    Arguments:
    - image (numpy.ndarray): The input image.
    - i (int): Row index of the patch.
    - j (int): Column index of the patch.
    - crop_size (int): Size of the patches.
    - overlap (int): Overlap between patches.

    Returns:
    - patch (numpy.ndarray): The extracted image patch.
    
    '''
    stride = crop_size - overlap
    start_y = i * stride
    start_x = j * stride
    patch = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return patch

def clean_mask(mask):
    '''
    Clean the mask by retaining only the largest connected component.
    Arguments:
    - mask (numpy.ndarray): The input binary mask.

    Returns:
    - cleaned_mask (numpy.ndarray): The cleaned binary mask with only the largest component.
    '''
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
    
def get_centroid(mask):
    '''
    Get the centroid of the largest connected component in the mask.
    Arguments:
    - mask (numpy.ndarray): The input binary mask.

    Returns:
    - centroid (tuple): The (row, column) coordinates of the centroid.
    '''
    labels = label(mask)
    regions = regionprops(labels)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    if len(regions)>0:
        return sorted_regions[0].centroid
    else:
        return (0,0)
    
def iou(mask1, mask2):
    '''
    Compute the Intersection over Union (IoU) between two binary masks.
    Arguments:
    - mask1 (numpy.ndarray): The first binary mask.
    - mask2 (numpy.ndarray): The second binary mask.

    Returns:
    - iou (float): The IoU value between the two masks.
    '''
    assert np.array_equal(mask1, mask1.astype(bool)), "mask1 is not binary"
    assert np.array_equal(mask2, mask2.astype(bool)), "mask2 is not binary"

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    iou = intersection / union if union != 0 else 0
    
    return iou
def compute_iou_shared(args):
    '''
    Compute IoU for a single centroid and update the shared dictionary.
    Arguments:
    - args (tuple): A tuple containing (i, centroid, mask, ids, mask_ious_dict, seg_masks_rs, seg_id).
    Returns:
    - None: Updates the shared dictionary in place.
    '''
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
    Arguments:
    - centroids (list): List of centroids for each segment.
    - mask (numpy.ndarray): The binary mask.
    - ids (numpy.ndarray): The array of segment IDs.
    - seg_masks_rs (numpy.ndarray): The resampled segment masks.
    - seg_ids (list): List of segment IDs.

    Returns:
    - mask_ious (numpy.ndarray): The array of IoU values for each segment ID
    '''

    with Manager() as manager:
        mask_ious_dict = manager.dict()

        data = [
            (i, centroids[i], mask, ids, mask_ious_dict, seg_masks_rs, seg_ids[i]) 
            for i in range(len(centroids))
        ]

        with Pool(cpu_count()) as pool:
            pool.map(compute_iou_shared, data)

        mask_ious = np.zeros_like(ids).astype(np.float64)
        for hit_id, iou_val in mask_ious_dict.items():
            mask_ious[ids == hit_id] = iou_val

        return mask_ious
    
def save_mask_as_geotiff(source_tif_path, mask, output_tif_path):
    '''
    Save the mask as a compressed GeoTIFF file, preserving geospatial metadata from the source TIFF.
    Arguments:
    - source_tif_path (str): Path to the source GeoTIFF file.
    - mask (numpy.ndarray): The mask to be saved.
    - output_tif_path (str): Path to save the output GeoTIFF file.

    Returns:
    - None: Saves the mask as a GeoTIFF file.
    '''
    with TiffFile(source_tif_path) as tif:
        original_tags = tif.pages[0].tags
        original_page = tif.pages[0]

        extratags = [
            (tag.code, tag.dtype, tag.count, tag.value, False)
            for tag in original_tags.values()
        ]

        geokeys = original_page.geokeys if hasattr(original_page, 'geokeys') else None
        resolution = original_page.tags.get('XResolution', None)
        resolution_unit = original_page.tags.get('ResolutionUnit', None)

    imwrite(
        output_tif_path,
        mask, 
        compression='deflate',
        metadata=None, 
        extratags=extratags,
        resolution=resolution.value if resolution else None,
        resolutionunit=resolution_unit.value if resolution_unit else None,
        photometric='minisblack'
    )

    print(f"Saved masks as compressed GeoTIFF to: {output_tif_path}")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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

def compute_mean(image, labeled_mask, labels):
    """Compute mean values for R, G, B channels for each label."""
    mean_r = ndi.labeled_comprehension(image[..., 0], labeled_mask, labels, np.mean, float, 0)
    mean_g = ndi.labeled_comprehension(image[..., 1], labeled_mask, labels, np.mean, float, 0)
    mean_b = ndi.labeled_comprehension(image[..., 2], labeled_mask, labels, np.mean, float, 0)

    return pd.DataFrame({'label': labels, 'mean_R': mean_r, 'mean_G': mean_g, 'mean_B': mean_b})

def compute_median(image, labeled_mask, labels):
    """Compute median values for R, G, B channels for each label."""
    median_r = ndi.labeled_comprehension(image[..., 0], labeled_mask, labels, np.median, float, 0)
    median_g = ndi.labeled_comprehension(image[..., 1], labeled_mask, labels, np.median, float, 0)
    median_b = ndi.labeled_comprehension(image[..., 2], labeled_mask, labels, np.median, float, 0)

    return pd.DataFrame({'label': labels, 'median_R': median_r, 'median_G': median_g, 'median_B': median_b})



def get_props_df(image, labeled_mask, resample=1, res=1):
    """Extract region properties and combine them with mean & median color values.
    Arguments:
    - image (np.ndarray): Input image of shape (H, W, C).
    - labeled_mask (np.ndarray): Labeled mask of shape (H, W) with unique labels for each region.
    - resample (int): Resampling factor for the labeled mask.
    - res (int): Resolution factor for the properties.
    
    Returns:
    - props_df (pd.DataFrame): DataFrame containing region properties and color statistics.
    """
    labeled = label(labeled_mask, background=0)
    
    props = regionprops_table(
        labeled,
        properties=('label', 'centroid', 'axis_major_length', 'axis_minor_length', 'area', 'perimeter'),
    )
    props_df = pd.DataFrame(props)

    props_df['axis_major_length'] = (props_df['axis_major_length'] / resample) * res
    props_df['axis_minor_length'] = (props_df['axis_minor_length'] / resample) * res
    props_df['area'] = (props_df['area'] / (resample ** 2)) * (res ** 2)
    props_df['perimeter'] = (props_df['perimeter'] / resample) * res

    # Compute shape-based indices
    props_df['IR'] = (4 * np.pi * props_df['area']) / (props_df['perimeter'] ** 2)
    props_df['h'] = ((props_df['axis_major_length'] - props_df['axis_minor_length']) ** 2) / \
                    ((props_df['axis_major_length'] + props_df['axis_minor_length']) ** 2)
    props_df['IRt'] = (4 * np.pi * (np.pi * props_df['axis_major_length'] * props_df['axis_minor_length'])) / \
                      (np.pi * (props_df['axis_major_length'] + props_df['axis_minor_length']) * 
                      (1 + ((3 * props_df['h']) / (10 + np.sqrt(4 - 3 * props_df['h']))))) ** 2
    props_df['IRn'] = props_df['IR'] / props_df['IRt']

    # Get the correct labels from props_df
    labels = props_df['label'].values

    # Compute mean and median values
    labeled=cv2.resize(labeled.astype(np.uint16), image.shape[:-1][::-1], interpolation = cv2.INTER_NEAREST)
    means_df = compute_mean(image, labeled, labels)
    medians_df = compute_median(image, labeled, labels)

    # Merge all data into props_df using 'label' to ensure correct alignment
    props_df = props_df.merge(means_df, on='label', how='left')
    props_df = props_df.merge(medians_df, on='label', how='left')

    return props_df


def accuracy(seg_masks, mask):
    '''
    Assess the accuracy of segmentation masks against ground truth masks.
    Parameters:
    - file_pth(str): Path to the directory containing the segmentation results.
    - mask(bool): Ground truth mask.
    
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
    #try:
    #    with open(os.path.join(file_pth, 'para.json'), 'r') as json_file:
    #        para = json.load(json_file)[0]
    #except:
    #    with open(os.path.join(file_pth, 'init_para.json'), 'r') as json_file:
    #        para = json.load(json_file)[0]
    #OutDIR=para.get('OutDIR')
    #DataDIR=para.get('DataDIR')
    #DSname=para.get('DatasetName')
    #resample_factor=para.get('resample_factor')
    #fid=para.get('fid')
    #image=load_image(DataDIR,DSname,fid)
    #if resample_factor!=1:
    #    pre_para={'Resample': {'fxy':resample_factor},
    #        }

    #    image=preprocessing_roulette(image, pre_para)
    #    print('resampled to: ', image.shape)

    #n_pass=len(os.listdir(os.path.join(file_pth,'Merged')))

    #seg_masks=np.array(np.load(os.path.join(file_pth,'Merged',f'Merged_Layers_{n_pass-1:03}.npy'), allow_pickle=True))

    #print('Mask imported from '+OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy')
    #print('masks size:', seg_masks.shape)
    print(len(np.unique(seg_masks))-1,' mask(s) loaded')
    print('No. of ground truth objects: '+str(len(np.unique(mask))-1))

    seg_masks=resample_fnc(seg_masks.astype(np.uint16),{'target_size':mask.shape[::-1], 'method':'nearest'})
    

    seg_ids=np.unique(seg_masks)
    centroids=[get_centroid(seg_masks==id) for id in seg_ids]
    #centroids=np.array(centroids)/resample_factor

    ids, counts=np.unique(mask, return_counts=True)
    ids, counts = ids[1:], counts[1:]
    area = counts
    ids = ids[np.argsort(area)]
    area = np.sort(area)

    point_based_ac=np.zeros_like(ids)
    seg_fp= np.zeros_like(seg_ids)
    for c in range(len(centroids))[1:]:
        hit_id=int(mask[int(centroids[c][0]),int(centroids[c][1])])
        point_based_ac[ids==hit_id]+=1
        if hit_id!=0:
                seg_fp[c]+=1
    mask_ious = update_mask_ious_shared(centroids[1:], mask, ids, seg_masks, seg_ids[1:])

    print('Mean mask IoU: ')
    print(np.mean(np.abs(mask_ious)))
    outdic={'point based':point_based_ac, 'iou':mask_ious
                , 'area':area, 'segment area':np.unique(seg_masks,return_counts=True)[1][1:]
                , 'segment hit':seg_fp[1:],'label_count':len(np.unique(mask))-1
                ,'mask_count':len(np.unique(seg_masks))-1}

    return outdic

def graph_coloring(label, k=500):
    '''
    Colors the labeled regions in the input label image such that no two adjacent regions share the same color.
    Arguments:
    label (np.array): 2D array of labeled regions.
    k (int): Number of nearest neighbors to consider for adjacency.
    
    Returns:
    new_label (np.array): 2D array of labeled regions with colors assigned.
    '''
    new_label = np.zeros(label.shape)
    new_label[np.isnan(label)] = np.nan
    contour_label = label.copy()
    contour_label[ndi.binary_erosion(~np.isnan(label))] = np.nan
    y, x = np.nonzero(~np.isnan(contour_label))
    la = label[y, x]
    pt = np.c_[x, y]
    tr = kdtree(pt)
    # construct neighborhood graph
    graph = {}
    for li in range(1, int(la.max()) + 1):
        _, ii = tr.query(pt[la == li], k=k, workers=-1)
        laii = la[ii]
        sl = np.argmax(laii != li, axis=1)
        laii = laii[np.arange(sl.shape[0]), sl]
        graph[li] = np.unique(laii).tolist()

    # graph coloring
    color = {}
    for li in graph:
        color_neighbors = {color[i] for i in graph[li] if i in color}
        c = 0
        while c in color_neighbors:
            c += 1
        color[li] = c

    colors = np.array(list(color.values()))
    for i, li in enumerate(colors):
        new_label[label == i+1] = li

    return new_label

def mod_ap(tp_area, fp_area):
    '''
    Calculate modified average precision (mod AP) based on true positive and false positive areas.
    Arguments:
    tp_area (np.array): Array of true positive areas.
    fp_area (np.array): Array of false positive areas.

    Returns:
    ap (float): The calculated modified average precision (mod AP).
    '''
    tps=np.ones_like(tp_area)
    fps=np.zeros_like(fp_area)
    all_area=np.hstack([tp_area,fp_area])
    ac=np.hstack([tps,fps])
    ac=ac[np.argsort(all_area)]
    for i in np.arange(1,len(ac)):
        ac[i]+=ac[i-1]

    precision=ac/np.arange(1,len(ac)+1)
    recall=ac/(len(ac))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap