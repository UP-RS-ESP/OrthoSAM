from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator_mod2 as SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import torch
import functions as fnc
from skimage.measure import label, regionprops
import time
from skimage.morphology import binary_dilation
import json
import sys
import subprocess
import cv2
import tqdm
import signal
from contextlib import redirect_stdout, redirect_stderr


def timeout_handler(signum, frame):
    raise TimeoutError

start_script = time.time()

OutDIR=sys.argv[1]
n_pass=int(sys.argv[2])
with open(OutDIR+f'/log.txt', 'a') as f:
    with redirect_stdout(f), redirect_stderr(f):
        print(f'---------------\n{n_pass} segment chunks\n\n')
        print('Loaded parameters from '+OutDIR)
        with open(OutDIR+'init_para.json', 'r') as json_file:
            init_para = json.load(json_file)[n_pass]
        print(init_para)
        resample_factor=init_para.get('1st_resample_factor')

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

        DataDIR=init_para.get('DataDIR')
        DSname=init_para.get('DatasetName')
        fid=init_para.get('fid')


        #defining clips
        n_pass_resample_factor=init_para.get('n_pass_resample_factor')
        crop_size=init_para.get('crop_size')
        last_resample_factor=init_para.get('resample_factor')
        dilation_size=init_para.get('dilation_size')
        min_pixel=(init_para.get('expected_min_size(sqmm)')/(init_para.get('resolution(mm)')**2))*resample_factor
        min_radi=init_para.get('min_radius')

        image=fnc.load_image(DataDIR,DSname,fid)
        org_shape=image.shape
        print('Image size:', image.shape)
        image=fnc.preprocessing_roulette(image, pre_para)

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
            
            if n_pass_resample_factor==1:#if resmpale factor not specified
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
            print(f'Third pass resample factor: {required_resampling}')
            with open(OutDIR+'init_para.json', 'r') as json_file:
                init_para_lst = json.load(json_file)
            init_para.update({'resample_factor': required_resampling})
            init_para_lst[n_pass]=init_para
            with open(OutDIR+f'init_para.json', 'w') as json_file:
                json.dump(init_para_lst, json_file, indent=4)

            print('Performing resampled first pass and second pass clipwise segmentation')
            subprocess.run(["python", first_second_run, OutDIR])
            print('Merging windows')
            subprocess.run(["python", Merging_window, OutDIR])

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
                ar_masks=fnc.clean_and_overwrite(ar_masks)

                plt.subplot(2,2,4)
                plt.imshow(image)
                #plt.imshow(ar_masks, cmap='nipy_spectral',alpha=0.6)
                plt.imshow(ar_masks>0,alpha=0.6)
                plt.axis('off')
                plt.title(f'Merged, total {len(np.unique(ar_masks))} mask(s)', fontsize=20)
                plt.tight_layout()
                plt.savefig(OutDIR+f'Merged_masks_withvoid_{n_pass:03}.png')
                plt.show()
                
                print(f'Saving id mask to '+OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy...')
                np.save(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy',ar_masks)
                print('Saved')

                cal_stats=False
                if cal_stats:
                    #calculate stats and save
                    print('Calculating stats')
                    stats=fnc.create_stats_df(ar_masks,resample_factor)
                    stats.to_hdf(OutDIR+'stats_df_thirdpass.h5', key='df', mode='w')
                    print('Stats saved')

                    from scipy.stats import gaussian_kde
                    plt.figure(figsize=(16, 10))
                    plt.subplot(2,2,1)
                    plt.xscale('log')
                    data = stats['area']
                    kde = gaussian_kde(data)
                    x = np.linspace(min(data), max(data), 1000)
                    kde_values = kde(x)
                    plt.plot(x, kde_values)

                    plt.xlabel('Area (pixel)')
                    plt.ylabel('Density')
                    plt.title('Density Plot of Area')
                    plt.grid()

                    plt.subplot(2,2,2)
                    plt.xscale('log')
                    frequencies, bin_edges = np.histogram(data, bins=30)
                    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
                    plt.plot(bin_midpoints, frequencies)

                    plt.xlabel('Area (pixel)')
                    plt.ylabel('Frequency')
                    plt.title('Frequency Plot of Area')
                    plt.grid()

                    plt.subplot(2,2,3)
                    kde = gaussian_kde(data)
                    x = np.linspace(min(data), max(data), 1000)
                    kde_values = kde(x)
                    plt.plot(x, kde_values)

                    plt.xlabel('Area (pixel)')
                    plt.ylabel('Density')
                    plt.title('Density Plot of Area (nms)')
                    plt.grid()

                    plt.subplot(2,2,4)
                    frequencies, bin_edges = np.histogram(data, bins=30)
                    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
                    plt.plot(bin_midpoints, frequencies)

                    plt.xlabel('Area (pixel)')
                    plt.ylabel('Frequency')
                    plt.title('Frequency Plot of Area (nms)')
                    plt.grid()
                    plt.suptitle(f'Third_pass { len(stats)} object(s)')
                    plt.tight_layout()
                    plt.savefig(OutDIR+'size_distribution_thirdpass.png')
                    plt.show()

                    def plot_cdf(data, label, ls,log=False):
                        sorted_data = np.sort(data)
                        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
                        if log:
                            plt.xscale('log')
                        plt.plot(sorted_data, cdf, ls, label=label)

                    plt.figure(figsize=(16, 6))

                    plt.subplot(2, 2, 1)
                    plot_cdf(stats['area'], label='', ls='-')
                    plt.title('CDF of Area')
                    plt.xlabel('Area')
                    plt.ylabel('CDF')
                    plt.grid()

                    plt.subplot(2, 2, 2)
                    plot_cdf(stats['major axis length'], label='Major axis', ls='--')
                    plot_cdf(stats['minor axis length'], label='Minor axis', ls='-')
                    plt.title('CDF of Axis Length')
                    plt.xlabel('Axis Length')
                    plt.ylabel('CDF')
                    plt.grid()
                    plt.legend()

                    plt.subplot(2, 2, 3)
                    plot_cdf(stats['area'], label='Label guided SAM', ls='-', log=True)
                    plt.title('CDF of Log Area')
                    plt.xlabel('Area')
                    plt.ylabel('CDF')
                    plt.grid()

                    plt.subplot(2, 2, 4)
                    plot_cdf(stats['major axis length'], label='Major axis', ls='--', log=True)
                    plot_cdf(stats['minor axis length'], label='Minor axis', ls='-', log=True)
                    plt.title('CDF of Log Axis Length')
                    plt.xlabel('Axis Length')
                    plt.ylabel('CDF')
                    plt.grid()
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(OutDIR+'size_axis_distribution_thirdpass.png')
                    plt.show()
            else:
                print('Void(s) identified but no valid mask was found')
        else:
            print('No void found. Minimum size threshold or dilation parameter may need to be adjusted')
        end_script = time.time()
        print('script took: ', end_script-start_script)
        print('Output saved to '+OutDIR)
        print('---------------\n\n\n\n\n\n')
