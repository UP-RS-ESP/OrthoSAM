import numpy as np
import glob
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import load_image, preprocessing_roulette, resample_fnc, get_centroid, update_mask_ious_shared

def accuracy(file_pth,i):
    print('Assessing accuracy for: ', file_pth, ' ', i)
    fn = '/DATA/vito/output/'+file_pth+'_00_noshadow'+'/'
    #fn.sort()
    #fn_pth = fn[i]
    with open(os.path.join(fn,'para.json'), 'r') as json_file:
        para=json.load(json_file)[0]

    OutDIR=para.get('OutDIR')
    DataDIR=para.get('DataDIR')
    DSname=para.get('DatasetName')
    fid=para.get('fid')
    resample_factor=para.get('resample_factor')

    print(DataDIR)
    print(DSname)
    print(fid)
    image=load_image(DataDIR,DSname,fid)
    print('Image size:', image.shape)
    if resample_factor!=1:
        pre_para={'Resample': {'fxy':resample_factor},
            #'Gaussian': {'kernel size':3}
            #'CLAHE':{'clip limit':2}#,
            #'Downsample': {'fxy':4},
            #'Buffering': {'crop size': crop_size}
            }

        image=preprocessing_roulette(image, pre_para)
        print('resampled to: ', image.shape)

    n_pass=len(os.listdir(OutDIR+'/Merged'))

    #try:
    #seg_masks=np.array(np.load(OutDIR+'Third/all_mask_third_pass_id.npy', allow_pickle=True))
    #print('Mask imported from '+OutDIR+'Third/all_mask_third_pass_id.npy')
    #third=True
    #except:
    #seg_masks=np.array(np.load(OutDIR+f'Merged/Merged_Layers_{n_pass-1:03}.npy', allow_pickle=True))
    #third=False
    #print('Third pass mask not found, merged first-second pass mask imported instead')
    #print('Mask imported from '+OutDIR+f'Third/Merged_Layers_{n_pass:03}.npy')
        
    seg_masks=np.array(np.load(OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy', allow_pickle=True))

    third=n_pass
    print('Mask imported from '+OutDIR+f'/Merged/Merged_Layers_{n_pass-1:03}.npy')
    print('masks size:', seg_masks.shape)
    print(len(np.unique(seg_masks)),' mask(s) loaded')


    mask=(np.load(os.path.join(DataDIR,DSname,'msk.npy'))).astype(np.uint16)
    #shadow_mask=load_image(DataDIR,DSname[:-3]+'shd',fid).astype(np.uint16)
    #mask_dw=resample_fnc(mask,{'target_size':image.shape[:-1][::-1], 'method':'nearest'})
    #shadow_mask_dw=resample_fnc(shadow_mask,{'target_size':image.shape[:-1][::-1], 'method':'nearest'})
    seg_masks_rs=resample_fnc(seg_masks.astype(np.uint16),{'target_size':mask.shape[::-1], 'method':'nearest'})
    print('No. of actual objects: '+str(len(np.unique(mask))-1))
    #print('resampled shape: ', mask_dw.shape)

    seg_ids=np.unique(seg_masks)
    centroids=[get_centroid(seg_masks==id) for id in seg_ids]
    centroids=np.array(centroids)/resample_factor

    ids, counts=np.unique(mask, return_counts=True)
    ids, counts = ids[1:], counts[1:]
    area = counts * (0.2 * 0.2)
    ids = ids[np.argsort(area)]
    area = np.sort(area)

    #shadowed_ids,shadowed_area=np.unique(mask[shadow_mask<1], return_counts=True)
    completely_in_shadow=np.zeros_like(ids)
    #for id in ids:
    #    if id in shadowed_ids:
    #        if shadowed_area[shadowed_ids==id]/area[ids==id]>0.9:
    #            completely_in_shadow[ids==id]=1

    point_based_ac=np.zeros_like(ids)
    for c in range(len(centroids))[1:]:
        hit_id=int(mask[int(centroids[c][0]),int(centroids[c][1])])
        point_based_ac[ids==hit_id]+=1
    #mask_ious=np.zeros_like(ids).astype(np.float64)
    #for n in tqdm(range(len(centroids))[1:], 'Matching and calculate IoU', unit='objects'):
    #    hit_id=int(mask[int(centroids[n][0]),int(centroids[n][1])])
    #    point_based_ac[ids==hit_id]+=1
    #    current_iou=mask_ious[ids==hit_id]
    #    iou=iou(seg_masks_rs==n, mask==hit_id)
    #    if iou>current_iou:
    #        mask_ious[ids==hit_id]=iou
    mask_ious = update_mask_ious_shared(centroids[1:], mask, ids, seg_masks_rs, seg_ids[1:])

    print('Mean mask IoU: ')
    print(np.mean(np.abs(mask_ious)))
    np.save(os.path.join(DataDIR,DSname,f'point_based_ac_org.npy'), {'point based':point_based_ac, 'iou':mask_ious, 'area':area, 'segment area':np.unique(seg_masks,return_counts=True)[1][1:]/resample_factor,'label_count':len(np.unique(mask))-1,'mask_count':len(np.unique(seg_masks)),'Third pass': third, 'para':para, 'completely_in_shadow':completely_in_shadow})