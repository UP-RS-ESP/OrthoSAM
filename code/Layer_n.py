import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import time
from skimage.morphology import binary_dilation
import json
import cv2
import tqdm
import signal
import os
from Layer_0 import predict_tiles
from Merging import merge_chunks
from utility import load_image, preprocessing_roulette, clean_and_overwrite


def timeout_handler(signum, frame):
    raise TimeoutError

def predict_tiles_n(para_L, n_pass):
    start_script = time.time()
    
    para = para_L[n_pass]
    OutDIR=para.get('OutDIR')

    print(f'---------------\nLayer {n_pass}\n\n')
    print('\tLoaded parameters from '+OutDIR)

    print('\t',para)
    resample_factor=para.get('1st_resample_factor')

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

    DataDIR=para.get('DataDIR')
    DSname=para.get('DatasetName')
    fid=para.get('fid')
    plotting=para.get('Plotting')


    #defining clips
    n_pass_resample_factor=para.get('resample_factor')
    crop_size=para.get('tile_size')
    last_resample_factor=para_L[n_pass-1].get('resample_factor')
    
    min_pixel=(para.get('expected_min_size(sqmm)')/(para.get('resolution(mm)')**2))*resample_factor

    image=load_image(DataDIR,DSname,fid)
    org_shape=image.shape
    print('\tImage size:', image.shape)
    image=preprocessing_roulette(image, pre_para)

    print('\tPreprocessing finished')

    ar_masks=np.array(np.load(os.path.join(OutDIR,f'Merged/Merged_Layers_{n_pass-1:03}.npy'), allow_pickle=True))
    print('\t',len(np.unique(ar_masks)),' mask(s) loaded')    
    
    if n_pass_resample_factor=='Auto':#if resmpale factor not specified
        # void searching only when resampling factor is not specified
        dilation_size=para.get('dilation_size')
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(ar_masks>=1, kernel)
        no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)

        regions=regionprops(no_mask_area)
        if plotting:
            plt.imshow(no_mask_area, cmap='nipy_spectral')
        fxy=[]
        for i,region in tqdm.tqdm(enumerate(regions), 'Checking required resampling for each void', total=len(regions)):
            # take regions with large enough areas
            if (region.area > min_pixel):
                mask=np.array(no_mask_area==(i+1))
                y0, x0 =region.centroid
                minr, minc, maxr, maxc = region.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                width, length=np.max(bx)-np.min(bx),np.max(by)-np.min(by)
                if plotting:
                    plt.plot(x0, y0, '.g', markersize=15)
                    plt.plot(bx, by, '-b', linewidth=2.5)
                factor=int((np.max([length/resample_factor,width/resample_factor]))//(crop_size*0.5))+1
                if factor<(np.max(org_shape)*0.8)/(crop_size):
                    fxy.append(factor)
        
        if plotting:
            plt.title('Significant mask gaps')
            plt.savefig(os.path.join(OutDIR,f'gaps_{n_pass:03}.png'))
            plt.show()

        #if len(fxy)>0:
        print(f'\t{len(fxy)} valid void(s) found')
        if len(fxy)>0:
            required_resampling=1/(np.max(fxy))
        else:
            required_resampling=last_resample_factor/2
            print('\tNo valid voids were detected. Adjust parameter if needed.')
            print(f'\tWould you like to further downsampling by a factor of 2 and proceed? (Resampling factor: {required_resampling})')
            signal.signal(signal.SIGALRM, timeout_handler)

            def prompt_user(timeout=10):
                try:
                    signal.alarm(timeout)
                    response = input("yes/no: ")
                    signal.alarm(0)
                    return response.lower() in ['yes', 'y']
                except TimeoutError:
                    return True 

            if prompt_user():
                required_resampling=last_resample_factor/2
            else:
                print("\tYou chose no. Exiting script.")
                exit()
    else:
        required_resampling=n_pass_resample_factor
    print(f'\tResample factor: {required_resampling}')
    #with open(os.path.join(OutDIR,'para.json'), 'r') as json_file:
    #    para_lst = json.load(json_file)
    para.update({'resample_factor': required_resampling})
    para_L[n_pass]=para
    with open(os.path.join(OutDIR,'para.json'), 'w') as json_file:
        json.dump(para_L, json_file, indent=4)

    predict_tiles(para_L, n_pass)
    merge_chunks(para_L,n_pass)

    resampled_SAM=np.array(np.load(os.path.join(OutDIR,f'Merged/Merged_Layers_{n_pass:03}.npy'), allow_pickle=True))
    resampled_SAM=cv2.resize(resampled_SAM.astype(np.uint16), ar_masks.shape[::-1], interpolation = cv2.INTER_NEAREST)

    #finding mask that is only inside the void
    ids_in_void,counts_in_void=np.unique(resampled_SAM[(ar_masks==0).astype('bool')], return_counts=True)
    ids_total,counts_total=np.unique(resampled_SAM, return_counts=True)
    valid_ids=[]
    for id,count in tqdm.tqdm(zip(ids_in_void,counts_in_void), total=len(ids_in_void), unit='id'):
        if count>=((counts_total[ids_total==id])*0.8):#if at least 80% of the object is not masked in previous layers
            valid_ids.append(id)

    if len(valid_ids)!=0:
        
        largest_id=np.max(ar_masks)

        mask = np.isin(resampled_SAM, valid_ids)
        id_mask = np.where(mask, resampled_SAM, 0)
        print(f'\tLayer {n_pass:03} discovered {len(valid_ids)} new mask(s)')
        #id_mask[id_mask>0]+=largest_id
        if plotting:
            plt.figure(figsize=(20,20))
            plt.subplot(2,2,1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'RGB', fontsize=20)
            plt.subplot(2,2,2)
            plt.imshow(ar_masks, cmap='nipy_spectral')
            plt.axis('off')
            plt.title(f'Last layer result, {len(np.unique(ar_masks))} mask(s)', fontsize=20)
            plt.subplot(2,2,3)
            #plt.imshow(image)
            plt.imshow(resampled_SAM, cmap='nipy_spectral')
            plt.title(f'All extracted masks from current layer, resampled factor: {required_resampling}', fontsize=20)
            plt.axis('off')
            plt.subplot(2,2,3)
            #plt.imshow(image)
            plt.imshow(id_mask, cmap='nipy_spectral')
            plt.axis('off')
            plt.title(f'Selected mask to merge from current layer, no. of new mask: {len(valid_ids)}', fontsize=20)

        ids=np.unique(id_mask)
        ids=ids[ids>0]
        for id in ids:
            ar_masks[id_mask==id]=(id+largest_id)
        
        #clean and remove empty lable
        ar_masks=clean_and_overwrite(ar_masks)

        if plotting:
            plt.subplot(2,2,4)
            plt.imshow(ar_masks>0)
            plt.axis('off')
            plt.title(f'Layers merged, total {len(np.unique(ar_masks))} mask(s)', fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(OutDIR,f'Merged_Layers_{n_pass:03}_output.png'))
            plt.show()
        
        print('\tSaving id mask to '+os.path.join(OutDIR,f'Merged/Merged_Layers_{n_pass:03}.npy')+'...')
        np.save(os.path.join(OutDIR,f'Merged/Merged_Layers_{n_pass:03}.npy'),ar_masks)
        print('\tSaved')
        #else:
        #    print('\tVoid(s) identified but no valid mask was found')
    else:
        print('\tNo void found. Minimum size threshold or dilation parameter may need to be adjusted')
    end_script = time.time()
    print(f'\tscript took: {end_script-start_script:.2f} seconds')
    print('\tOutput saved to '+OutDIR)
    print('---------------\n\n\n\n\n\n')
#except Exception as e:
#    import traceback
#    traceback.print_exc()