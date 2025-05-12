import numpy as np
import matplotlib.pyplot as plt
import glob
import functions as fnc
import time
import sys
import json
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr

start_script = time.time()

OutDIR=sys.argv[1]

n_pass=len(os.listdir(OutDIR+'Merged'))
with open(OutDIR+f'/log.txt', 'a') as f:
    with redirect_stdout(f), redirect_stderr(f):
        try:
            print(f'---------------\n{n_pass} merging chunks\n\n')
            print('Loaded parameters from '+OutDIR)
            with open(OutDIR+'init_para.json', 'r') as json_file:
                init_para = json.load(json_file)[n_pass]

            resample_factor=init_para.get('resample_factor')

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

            print(init_para)
            print(pre_para)

            OutDIR=init_para.get('OutDIR')
            DataDIR=init_para.get('DataDIR')
            DSname=init_para.get('DatasetName')
            fid=init_para.get('fid')

            #defining clips
            b=init_para.get('b')
            crop_size=init_para.get('crop_size')

            image=fnc.load_image(DataDIR,DSname,fid)
            org_shape=image.shape
            print('Image size:', image.shape)

            if resample_factor!=1:
                image=fnc.preprocessing_roulette(image, pre_para)
                print('Resampled to: ', image.shape)

            print('Loading clips.....')

            clips_pths = glob.glob(OutDIR+f'chunks/{n_pass}/chunk_*')
            clips_pths.sort()


            print(len(clips_pths),' clips imported')

            msk_count=0
            id_mask = np.zeros_like(image[:,:,0], dtype=np.uint32)
            stack_mask = np.zeros_like(image[:,:,0], dtype=np.uint32)

            #Merging windows
            Aggregate_masks_noedge=[]
            pred_iou_noedge=[]
            for w_count,pth in tqdm(enumerate(clips_pths),f'Merging and resizing clips', total=len(clips_pths), unit='clips'):
                clip_window=np.load(pth, allow_pickle=True)[0]
                i,j=clip_window['ij']

                for mask,score in tqdm(zip(clip_window['nms mask'], clip_window['nms mask pred iou']), f'Merging and resizing masks in clip {i,j} (RAM: {fnc.get_memory_usage():.2f} MB, {msk_count} masks)',unit='masks',leave=False,total=len(clip_window['nms mask pred iou'])):
                    if not (np.any(mask[0]==1) or np.any(mask[-1]==1) or np.any(mask[:,0]==1) or np.any(mask[:,-1]==1)):
                        resized = fnc.untile(id_mask, mask, i, j, crop_size, 2*b)
                        msk_count+=1
                        id_mask[resized!=0]=(msk_count)
                        stack_mask+=resized
                clip_window.clear()


            #clean and remove empty lable
            id_mask=fnc.clean_and_overwrite(id_mask)

            print(f'Saving id mask to '+OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy...')
            np.save(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}',id_mask)
            print('Saved')

            print(f'{msk_count} masks found')

            plt.figure(figsize=(20,20))
            plt.subplot(2,2,1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'RGB', fontsize=20)
            plt.subplot(2,2,2)
            plt.imshow(image)
            plt.imshow(stack_mask,alpha=0.6)
            plt.axis('off')
            plt.title(f'Max overlap: {np.max(stack_mask)}', fontsize=20)
            plt.subplot(2,2,3)
            plt.imshow(image)
            plt.imshow(id_mask, cmap='nipy_spectral',alpha=0.5)
            plt.title(f'Segments, {msk_count} mask(s)', fontsize=20)
            plt.subplot(2,2,4)
            plt.imshow(image)
            plt.imshow(stack_mask>1,alpha=0.6)
            plt.axis('off')
            plt.title(f'Overlapping area', fontsize=20)
            plt.tight_layout()
            plt.savefig(OutDIR+f'Merged_mask_{n_pass:03}.png')
            plt.show()

            end_script = time.time()
            print('script took: ', end_script-start_script)
            print('Merging windows completed. Output saved to '+OutDIR)
            print('---------------\n\n\n\n\n\n')
        except Exception as e:
            import traceback
            traceback.print_exc()