import glob
import numpy as np
from tifffile import imread
import cv2
import torch
from skimage.measure import label, regionprops
import os
import json
from segment_anything import sam_model_registry
from torchvision.ops.boxes import batched_nms
import psutil
from tqdm import tqdm
import sys

def load_image(DataDIR,DSname,fid):
    if isinstance(fid, int):
        fn_img = glob.glob(os.path.join(DataDIR,DSname,'*'))
        print(fn_img)
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

def load_config(filename='config.json'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, filename)

    with open(config_path, 'r') as f:
        return json.load(f)
    
def set_sam(MODEL_TYPE,CheckpointDIR):
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
    process = psutil.Process()
    mem_info = process.memory_info()
    # Convert from bytes to MB
    return mem_info.rss / (1024 * 1024)

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

def clean_and_overwrite(mask):
    labeled = label(mask)
    id_to_remove = []
    for i in tqdm(np.unique(mask)[1:]):
        if len(np.unique(labeled[mask==i]))>1:
            labels, counts = np.unique(labeled[mask==i], return_counts=True)   
            max_label = labels[np.argmax(counts)]
            id_to_remove.append(labels[labels != max_label])
    if len(id_to_remove)>0:
        for i in np.hstack(id_to_remove):
            mask[labeled==i]=0
    return label(mask)

def create_dir_ifnotexist(OutDIR):
    if not os.path.exists(OutDIR):
        os.makedirs(OutDIR)
    if not os.path.exists(os.path.join(OutDIR,'chunks')):
        os.makedirs(os.path.join(OutDIR,'chunks'))
    if not os.path.exists(os.path.join(OutDIR,'Merged')):
        os.makedirs(os.path.join(OutDIR,'Merged'))

def prompt_fid(para):
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
    master_para['1st_resample_factor'] = master_para['resample_factor']
    config = load_config()
    master_para={**config,**master_para}
    master_para['DataDIR'] = config.get('DataDIR')
    if not os.path.exists(os.path.join(master_para.get('DataDIR'),master_para.get('DatasetName'))):
        print('Input directory does not exist. Exiting script.')
        sys.exit()

    # create dir if output dir not exist
    OutDIR=master_para.get('OutDIR')
    create_dir_ifnotexist(OutDIR)
    if master_para.get('fid')==None:
        master_para=prompt_fid(master_para)


    # Save para to a JSON file
    lst = [dict(master_para, **para) for para in para_list]
    with open(os.path.join(OutDIR,'para.json'), 'w') as json_file:
        json.dump(lst, json_file, indent=4)
    if pre_para_list:
        with open(os.path.join(OutDIR,'pre_para.json'), 'w') as json_file:
            json.dump(pre_para_list, json_file, indent=4)
    return OutDIR