import numpy as np
import matplotlib.pyplot as plt
import glob
from utility import load_image, preprocessing_roulette, untile, clean_and_overwrite, get_memory_usage
import time
import json
import os
from tqdm import tqdm
import sys

def merge_chunks(para_list, n_pass):
    start_script = time.time()
    
    para = para_list[n_pass]
    OutDIR=para.get('OutDIR')
    #try:
    print(f'\tMerging tiles\n\n')
    print('\tLoaded parameters from '+OutDIR)

    resample_factor=para.get('resample_factor')

    try:#attempt to load saved pre_para
        with open(os.path.join(OutDIR,'pre_para.json'), 'r') as json_file:
            pre_para = json.load(json_file)[n_pass]
        pre_para.update({'Resample': {'fxy':resample_factor}})
        print('\tLoaded preprocessing parameters from json')
        print('\t',pre_para)
    except:#use defined para
        print('\tNo pre_para found. Only applying resampling.')
        pre_para={'Resample': {'fxy':resample_factor}}
        print('\t',pre_para)

    OutDIR=para.get('OutDIR')
    DataDIR=para.get('DataDIR')
    DSname=para.get('DatasetName')
    fid=para.get('fid')
    plotting=para.get('Plotting')

    #defining clips
    b=para.get('tile_overlap')
    crop_size=para.get('tile_size')

    image=load_image(DataDIR,DSname,fid)
    #org_shape=image.shape
    print('\tImage size:', image.shape)

    if resample_factor!=1:
        image=preprocessing_roulette(image, pre_para)
        print('\tResampled to: ', image.shape)

    print('\tLoading tiles.....')

    clips_pths = glob.glob(os.path.join(OutDIR,f'chunks/{n_pass}/chunk_*'))
    clips_pths.sort()


    print('\t',len(clips_pths),' tiles imported')

    msk_count=0
    id_mask = np.zeros_like(image[:,:,0], dtype=np.uint32)
    stack_mask = np.zeros_like(image[:,:,0], dtype=np.uint32)

    #Merging windows
    Aggregate_masks_noedge=[]
    pred_iou_noedge=[]
    for w_count,pth in tqdm(enumerate(clips_pths),f'Merging and resizing tiles', total=len(clips_pths), unit='tiles', file=sys.stdout):
        clip_window=np.load(pth, allow_pickle=True)[0]
        i,j=clip_window['ij']

        for mask,score in tqdm(zip(clip_window['nms mask'], clip_window['nms mask pred iou']), f'Merging and resizing masks in tile {i,j} (RAM: {get_memory_usage():.2f} MB, {msk_count} masks)',unit='masks',leave=False,total=len(clip_window['nms mask pred iou']), file=sys.stdout):
            if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                resized = untile(id_mask, mask, i, j, crop_size, b)
                msk_count+=1
                id_mask[resized!=0]=(msk_count)
                stack_mask+=resized
        clip_window.clear()


    #clean and remove empty lable
    id_mask=clean_and_overwrite(id_mask)

    print(f'\tSaving id mask to '+OutDIR+f'Merged/Merged_Layers_{n_pass:03}.npy...')
    np.save(os.path.join(OutDIR,f'Merged/Merged_Layers_{n_pass:03}'),id_mask)
    print('\tSaved')

    print(f'{msk_count} masks found')
    if plotting:
        plt.figure(figsize=(15,8))
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'RGB', fontsize=20)
        plt.subplot(1,3,2)
        #plt.imshow(image)
        #plt.imshow(stack_mask,alpha=0.6)
        plt.axis('off')
        #plt.title(f'Max overlap: {np.max(stack_mask)}', fontsize=20)
        #plt.subplot(1,3,3)
        #plt.imshow(image)
        plt.imshow(id_mask, cmap='nipy_spectral')
        plt.title(f'Segments id, {msk_count} mask(s)', fontsize=20)
        plt.subplot(1,3,3)
        #plt.imshow(image)
        plt.imshow(id_mask>1)
        plt.axis('off')
        plt.title(f'Segmented', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OutDIR,f'Layer_{n_pass:03}_output.png'))
        plt.show()

    end_script = time.time()
    print(f'\tscript took: {end_script-start_script:.2f} seconds')
    print('\tMerging completed.')
    print('---------------\n\n\n\n\n\n')
    #except Exception as e:
    #    import traceback
    #    traceback.print_exc()