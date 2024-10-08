import torch
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator_mod2 as SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import torch
import functions as fnc
from importlib import reload
import gc
from skimage.measure import label, regionprops
from torchvision.ops.boxes import batched_nms
from sklearn.neighbors import KDTree
import time
from skimage.morphology import binary_dilation
import json
import sys
import subprocess
import cv2
import tqdm

start_script = time.time()

try:
    OutDIR=sys.argv[1]
except:
    OutDIR='/DATA/vito/output/Ravi3_run2_dw2_stb085_3b_minarea/'

print('Loaded parameters from '+OutDIR)
with open(OutDIR+'init_para.json', 'r') as json_file:
    init_para = json.load(json_file)
with open(OutDIR+'pre_para.json', 'r') as json_file:
    pre_para = json.load(json_file)

print(init_para)
print(pre_para)

DataDIR=init_para.get('DataDIR')
DSname=init_para.get('DatasetName')
fid=init_para.get('fid')


#defining clips
third_b_resmpale=init_para.get('third_b_resample_factor')
crop_size=init_para.get('crop_size')
resample_factor=init_para.get('resample_factor')
dilation_size=init_para.get('dilation_size')
min_pixel=(init_para.get('resolution(mm)')**2)/init_para.get('expected_min_size(sqmm)')
min_radi=init_para.get('min_radius')

image=fnc.load_image(DataDIR,DSname,fid)
org_shape=image.shape
print('Image size:', image.shape)
image=fnc.preprocessing_roulette(image, pre_para)

print('Preprocessing finished')

ar_masks=np.array(np.load(OutDIR+'Merged/all_mask_merged_windows_id.npy', allow_pickle=True))
print(len(np.unique(ar_masks)),' mask(s) loaded')

# Define the paths to the scripts you want to run
first_second_run = "code/First_second_pass_newtile.py"
Merging_window = "code/Merging_window_newtile.py"

#identify void
kernel = np.ones((dilation_size, dilation_size), np.uint8)
stacked_Aggregate_masks_noedge_nms_eroded=binary_dilation(ar_masks>=1, kernel)
no_mask_area=label(stacked_Aggregate_masks_noedge_nms_eroded,1,False,1)

regions=regionprops(no_mask_area)

list_of_no_mask_area_centroid=[]
list_of_no_mask_area_mask=[]
list_of_no_mask_area_bbox=[]

for i,region in tqdm.tqdm(enumerate(regions), total=len(regions)):
    # take regions with large enough areas
    if (region.area > min_pixel):
        mask=np.array(no_mask_area==(i+1))
        y0, x0 =region.centroid
        minr, minc, maxr, maxc = region.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        list_of_no_mask_area_centroid.append((y0,x0))
        list_of_no_mask_area_mask.append(no_mask_area==(i+1))
        list_of_no_mask_area_bbox.append([bx,by])

for i,region in tqdm.tqdm(enumerate(regions), total=len(regions)):
    # take regions with large enough areas
    if (region.area > min_pixel):
        mask=np.array(no_mask_area==(i+1))
        y0, x0 =region.centroid
        minr, minc, maxr, maxc = region.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        list_of_no_mask_area_centroid.append((y0,x0))
        list_of_no_mask_area_mask.append(no_mask_area==(i+1))
        list_of_no_mask_area_bbox.append([bx,by])

if len(regions)>0:
    print(f'{len(list_of_no_mask_area_mask)} void(s) found')
    ar_no_mask_area=np.stack(list_of_no_mask_area_mask)
    stacked=np.sum(ar_no_mask_area,axis=0)
    plt.imshow(stacked, cmap='nipy_spectral')
    fxy=[]
    for i in range(ar_no_mask_area.shape[0]):
        y0, x0 =list_of_no_mask_area_centroid[i]
        bx, by =list_of_no_mask_area_bbox[i]

        plt.plot(x0, y0, '.g', markersize=15)
        plt.plot(bx, by, '-b', linewidth=2.5)
        width, length=np.max(bx)-np.min(bx),np.max(by)-np.min(by)
        factor=int((np.max([length/resample_factor,width/resample_factor]))//(crop_size*0.6))+1
        if factor<(np.max(org_shape)*0.8)/(crop_size):
            fxy.append(factor)
    plt.title('Void in clean SAM masks')
    plt.savefig(OutDIR+'void.png')
    plt.show()

    if third_b_resmpale==1:#if resmpale factor not specified
        if len(fxy)>0:
            required_resampling=np.max(fxy)
        else:
            required_resampling=resample_factor/2
    else:
        required_resampling=third_b_resmpale
    print(f'Third pass resample factor: {required_resampling}')
    
    third_init_para=init_para.copy()
    third_init_para.update({'OutDIR': OutDIR+'Third/',
                            'resample_factor': 1/required_resampling})
    third_pre_para=pre_para.copy()
    third_pre_para.update({'Resample': {'fxy': 1/required_resampling}})

    fnc.create_dir_ifnotexist(OutDIR+'Third/')
    with open(OutDIR+'Third/'+'init_para.json', 'w') as json_file:
        json.dump(third_init_para, json_file, indent=4)
    with open(OutDIR+'Third/'+'pre_para.json', 'w') as json_file:
        json.dump(third_pre_para, json_file, indent=4)
    print('Performing resampled first pass and second pass clipwise segmentation')
    subprocess.run(["python", first_second_run, OutDIR+'Third/'])
    print('Merging windows')
    subprocess.run(["python", Merging_window, OutDIR+'Third/'])

    resampled_SAM=np.array(np.load(OutDIR+'Third/'+'Merged/all_mask_merged_windows_id.npy', allow_pickle=True))
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
        print(f'Third pass discovered {len(valid_ids)} new mask(s)')
        #id_mask[id_mask>0]+=largest_id
        
        plt.figure(figsize=(20,20))
        plt.subplot(2,2,1)
        plt.imshow(image)
        plt.imshow(ar_masks,alpha=0.6)
        plt.axis('off')
        plt.title(f'First and second pass result, {len(np.unique(ar_masks))} mask(s)', fontsize=20)
        plt.subplot(2,2,2)
        plt.imshow(image)
        plt.imshow((ar_masks+id_mask)>0,alpha=0.6)
        plt.axis('off')
        plt.title(f'Masked area with third pass, resampled factor: {1/required_resampling}', fontsize=20)
        plt.subplot(2,2,3)
        plt.imshow(ar_masks+id_mask, cmap='nipy_spectral')
        plt.axis('off')
        plt.title(f'Mask area with third pass\n No. of mask: {len(np.unique(id_mask))+len(np.unique(ar_masks))}', fontsize=20)
        plt.subplot(2,2,4)
        plt.imshow(image)
        plt.imshow(((id_mask!=0)+(ar_masks!=0))>1,alpha=0.6)
        plt.axis('off')
        plt.title(f'Overlapping area, Overlap: {np.sum((id_mask!=0)+(ar_masks!=0)>1)}', fontsize=20)
        plt.tight_layout()
        plt.savefig(OutDIR+'Merged_masks_withvoid.png')
        plt.show()

        ids=np.unique(id_mask)
        ids=ids[ids>0]
        for id in ids:
            ar_masks[id_mask==id]=(id+largest_id)
        
        #shuffle id
        unique_labels = np.unique(ar_masks)
        if 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]
        shuffled_labels = np.random.permutation(unique_labels)
        label_mapping = dict(zip(unique_labels, shuffled_labels))
        shuffled_mask = ar_masks.copy()
        for old_label, new_label in label_mapping.items():
            shuffled_mask[ar_masks == old_label] = new_label
        ar_masks=shuffled_mask
        
        print(f'Saving id mask to '+OutDIR+'Third/all_mask_merged_windows_id.npy...')
        np.save(OutDIR+'Third/all_mask_thid_pass_id',ar_masks)
        print('Saved')

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
        plt.suptitle(f'Thid_pass { len(stats)} object(s)')
        plt.tight_layout()
        plt.savefig(OutDIR+'size_distribution_thidpass.png')
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
        plt.savefig(OutDIR+'size_axis_distribution_thidpass.png')
        plt.show()
    else:
        print('Void(s) identified but no valid mask was found')
else:
    print('No void found. Minimum size threshold or dilation parameter may need to be adjusted')
end_script = time.time()
print('script took: ', end_script-start_script)
print('Third pass SAM completed. Output saved to '+OutDIR)
