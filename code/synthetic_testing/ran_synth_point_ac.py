import numpy as np
import glob
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functions as fnc
from tqdm import tqdm


def accuracy(file_pth):
    fn = glob.glob('/DATA/vito/output/'+file_pth+'/*')
    fn.sort()
    para_list=[]
    for fn_pth in fn:
        with open(fn_pth+'/para.json', 'r') as json_file:
            para_list.append(json.load(json_file)[0])

    print(fn)


    for i,init_para in enumerate(para_list): 

        OutDIR=init_para.get('OutDIR')
        DataDIR=init_para.get('DataDIR')
        DSname=init_para.get('DatasetName')
        fid=init_para.get('fid')
        resample_factor=init_para.get('resample_factor')
        b=init_para.get('b')
        crop_size=init_para.get('crop_size')

        image=fnc.load_image(DataDIR,DSname,fid)
        print('Image size:', image.shape)
        if resample_factor!=1:
            pre_para={'Resample': {'fxy':resample_factor},
                #'Gaussian': {'kernel size':3}
                #'CLAHE':{'clip limit':2}#,
                #'Downsample': {'fxy':4},
                #'Buffering': {'crop size': crop_size}
                }

            image=fnc.preprocessing_roulette(image, pre_para)
            print('resampled to: ', image.shape)

        n_pass=len(os.listdir(OutDIR+'/Merged'))

        #try:
        #seg_masks=np.array(np.load(OutDIR+'Third/all_mask_third_pass_id.npy', allow_pickle=True))
        #print('Mask imported from '+OutDIR+'Third/all_mask_third_pass_id.npy')
        #third=True
        #except:
        #seg_masks=np.array(np.load(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass-1:03}.npy', allow_pickle=True))
        #third=False
        #print('Third pass mask not found, merged first-second pass mask imported instead')
        #print('Mask imported from '+OutDIR+f'Third/all_mask_merged_windows_id_{n_pass:03}.npy')
        if n_pass==0:
            seg_masks=np.array(np.load(OutDIR+f'/Merged/Merged_Layers_{n_pass:03}.npy', allow_pickle=True))
            print('Mask imported from '+OutDIR+f'/Merged/Merged_Layers_{n_pass:03}.npy')
        else:
            seg_masks=np.array(np.load(OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy', allow_pickle=True))
            print('Mask imported from '+OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy')
        #if n_pass==0:
        #    seg_masks=np.array(np.load(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass:03}.npy', allow_pickle=True))
        #    print('Mask imported from '+OutDIR+f'/Merged/all_mask_merged_windows_id_{n_pass:03}.npy')
        #else:
        #    seg_masks=np.array(np.load(OutDIR+f'Merged/all_mask_merged_windows_id_{n_pass-1:03}.npy', allow_pickle=True))
        #    print('Mask imported from '+OutDIR+f'/Merged/all_mask_merged_windows_id_{n_pass-1:03}.npy')
        third=n_pass
        
        print('masks size:', seg_masks.shape)
        print(len(np.unique(seg_masks)),' mask(s) loaded')

        fn_img = glob.glob(DataDIR+'/'+DSname[:-3]+'msk/*')
        fn_img.sort()
        mask=(np.load(fn_img[fid])).astype(np.uint16)
        mask_dw=fnc.resample_fnc(mask,{'target_size':image.shape[:-1][::-1], 'method':'nearest'})
        seg_masks_rs=fnc.resample_fnc(seg_masks.astype(np.uint16),{'target_size':mask.shape[::-1], 'method':'nearest'})
        print(fn_img[fid].split("/")[-1]+' imported')
        print('No. of actual objects: '+str(len(np.unique(mask))-1))
        print('resampled shape: ', mask_dw.shape)

        seg_ids=np.unique(seg_masks)
        centroids=[fnc.get_centroid(seg_masks==id) for id in seg_ids]
        centroids=np.array(centroids)/resample_factor

        ids, counts=np.unique(mask, return_counts=True)
        ids, counts = ids[1:], counts[1:]
        area = counts * (0.2 * 0.2)
        ids = ids[np.argsort(area)]
        area = np.sort(area)

        point_based_ac=np.zeros_like(ids)
        seg_fp= np.zeros_like(seg_ids)
        for c in range(len(centroids))[1:]:
            hit_id=int(mask[int(centroids[c][0]),int(centroids[c][1])])
            point_based_ac[ids==hit_id]+=1
            if hit_id!=0:
                seg_fp[c]+=1
        #mask_ious=np.zeros_like(ids).astype(np.float64)
        #for n in tqdm(range(len(centroids))[1:], 'Matching and calculate IoU', unit='objects'):
        #    hit_id=int(mask[int(centroids[n][0]),int(centroids[n][1])])
        #    point_based_ac[ids==hit_id]+=1
        #    current_iou=mask_ious[ids==hit_id]
        #    iou=fnc.iou(seg_masks_rs==n, mask==hit_id)
        #    if iou>current_iou:
        #        mask_ious[ids==hit_id]=iou
        mask_ious = fnc.update_mask_ious_shared(centroids[1:], mask, ids, seg_masks_rs, seg_ids[1:])

        print('Mean mask IoU: ')
        print(np.mean(np.abs(mask_ious)))
        np.save(DataDIR+'/'+DSname[:-3]+file_pth+f'/point_based_ac_{i:02}.npy', {'point based':point_based_ac, 'iou':mask_ious, 'area':area, 'segment area':np.unique(seg_masks,return_counts=True)[1][1:]/resample_factor, 'segment hit':seg_fp[1:],'label_count':len(np.unique(mask))-1,'mask_count':len(np.unique(seg_masks)),'Third pass': third, 'init_para':init_para})