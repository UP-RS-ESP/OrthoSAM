import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import time
from skimage.morphology import binary_dilation
import json
import cv2
import tqdm
import signal
from Layer_0 import predict_tiles
from Merging import merge_chunks
from utility import load_image, preprocessing_roulette, clean_and_overwrite


def timeout_handler(signum, frame):
    raise TimeoutError

def predict_tiles_n(OutDIR, n_pass):
        #try: 
        start_script = time.time()
        print(f'---------------\n{n_pass} segment chunks\n\n')
        print('Loaded parameters from '+OutDIR)
        with open(OutDIR+'para.json', 'r') as json_file:
            para_L = json.load(json_file)
        para = para_L[n_pass]
        print(para)
        resample_factor=para.get('1st_resample_factor')

        try:#attempt to load saved pre_para
            with open(OutDIR+'pre_para.json', 'r') as json_file:
                pre_para = json.load(json_file)[n_pass]
            pre_para.update({'Resample': {'fxy':resample_factor}})
            print('Loaded preprocessing parameters from json')
            print(pre_para)
        except:#use defined para
            print('No pre_para found. Only applying resampling.')
            pre_para={'Resample': {'fxy':resample_factor}}
            print(pre_para)

        DataDIR=para.get('DataDIR')
        DSname=para.get('DatasetName')
        fid=para.get('fid')


        #defining clips
        n_pass_resample_factor=para.get('n_pass_resample_factor')
        crop_size=para.get('crop_size')
        last_resample_factor=para_L[n_pass-1].get('resample_factor')
        dilation_size=para.get('dilation_size')
        min_pixel=(para.get('expected_min_size(sqmm)')/(para.get('resolution(mm)')**2))*resample_factor
        min_radi=para.get('min_radius')

        image=load_image(DataDIR,DSname,fid)
        org_shape=image.shape
        print('Image size:', image.shape)
        image=preprocessing_roulette(image, pre_para)

        print('Preprocessing finished')

        ar_masks=np.array(np.load(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass-1:03}.npy', allow_pickle=True))
        print(len(np.unique(ar_masks)),' mask(s) loaded')

        first_second_run = "code/First_second_pass_newtile.py"
        Merging_window = "code/Merging_window_newtile.py"

        #identify void
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(ar_masks>=1, kernel)
        no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)

        regions=regionprops(no_mask_area)

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

                plt.plot(x0, y0, '.g', markersize=15)
                plt.plot(bx, by, '-b', linewidth=2.5)
                factor=int((np.max([length/resample_factor,width/resample_factor]))//(crop_size*0.5))+1
                if factor<(np.max(org_shape)*0.8)/(crop_size):
                    fxy.append(factor)
        plt.title('Void in clean SAM masks')
        plt.savefig(OutDIR+f'void_{n_pass:03}.png')
        plt.show()

        if len(fxy)>0:
            print(f'{len(fxy)} void(s) found')
            
            if n_pass_resample_factor=='Auto':#if resmpale factor not specified
                if len(fxy)>0:
                    required_resampling=1/(np.max(fxy))
                else:
                    required_resampling=last_resample_factor/2
                    print('No valid voids were detected. Adjust parameter if needed.')
                    print(f'Would you like to further downsampling by a factor of 2 and proceed? (Resampling factor: {required_resampling})')
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
                        print("You chose no. Exiting script.")
                        exit()
            else:
                required_resampling=n_pass_resample_factor
            print(f'Resample factor: {required_resampling}')
            with open(OutDIR+'para.json', 'r') as json_file:
                para_lst = json.load(json_file)
            para.update({'resample_factor': required_resampling})
            para_lst[n_pass]=para
            with open(OutDIR+f'para.json', 'w') as json_file:
                json.dump(para_lst, json_file, indent=4)

            predict_tiles(OutDIR, n_pass)
            merge_chunks(OutDIR,n_pass)

            resampled_SAM=np.array(np.load(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy', allow_pickle=True))
            resampled_SAM=cv2.resize(resampled_SAM.astype(np.uint16), ar_masks.shape[::-1], interpolation = cv2.INTER_NEAREST)

            #finding mask that is only inside the void
            ids_in_void,counts_in_void=np.unique(resampled_SAM[(ar_masks==0).astype('bool')], return_counts=True)
            ids_total,counts_total=np.unique(resampled_SAM, return_counts=True)
            valid_ids=[]
            for id,count in tqdm.tqdm(zip(ids_in_void,counts_in_void), total=len(ids_in_void), unit='id'):
                if count>=((counts_total[ids_total==id])*0.85):#if the object is at least 85% inside the voids
                    valid_ids.append(id)

            if len(valid_ids)!=0:
                
                largest_id=np.max(ar_masks)

                mask = np.isin(resampled_SAM, valid_ids)
                id_mask = np.where(mask, resampled_SAM, 0)
                print(f'{n_pass:03} pass discovered {len(valid_ids)} new mask(s)')
                #id_mask[id_mask>0]+=largest_id
                
                plt.figure(figsize=(20,20))
                plt.subplot(2,2,1)
                plt.imshow(image)
                plt.imshow(ar_masks,alpha=0.6)
                plt.axis('off')
                plt.title(f'{n_pass-1:03} pass result, {len(np.unique(ar_masks))} mask(s)', fontsize=20)
                plt.subplot(2,2,2)
                plt.imshow(image)
                plt.imshow(resampled_SAM,alpha=0.6)
                plt.title(f'All extracted masks from {n_pass:03} pass, resampled factor: {required_resampling}', fontsize=20)
                
                plt.axis('off')
                plt.subplot(2,2,3)
                plt.imshow(image)
                plt.imshow(id_mask,alpha=0.6)
                plt.axis('off')
                plt.title(f'{n_pass:03} pass extracted masks, no. of new mask: {len(valid_ids)}', fontsize=20)
                

                ids=np.unique(id_mask)
                ids=ids[ids>0]
                for id in ids:
                    ar_masks[id_mask==id]=(id+largest_id)
                
                #clean and remove empty lable
                ar_masks=clean_and_overwrite(ar_masks)

                plt.subplot(2,2,4)
                plt.imshow(image)
                plt.imshow(ar_masks>0,alpha=0.6)
                plt.axis('off')
                plt.title(f'Merged, total {len(np.unique(ar_masks))} mask(s)', fontsize=20)
                plt.tight_layout()
                plt.savefig(OutDIR+f'Merged_masks_withvoid_{n_pass:03}.png')
                plt.show()
                
                print(f'Saving id mask to '+OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy...')
                np.save(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy',ar_masks)
                print('Saved')
            else:
                print('Void(s) identified but no valid mask was found')
        else:
            print('No void found. Minimum size threshold or dilation parameter may need to be adjusted')
        end_script = time.time()
        print('script took: ', end_script-start_script)
        print('Output saved to '+OutDIR)
        print('---------------\n\n\n\n\n\n')
    #except Exception as e:
    #    import traceback
    #    traceback.print_exc()