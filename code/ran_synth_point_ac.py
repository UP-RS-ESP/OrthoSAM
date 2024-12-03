import numpy as np
import glob
import functions as fnc

para_list=[]
for i in range(10):
      para_list.append({'OutDIR': f'/DATA/vito/output/ran_synth/ran_synth_{i:02}_dw4_cp1024_3b/',
                        'DataDIR': '/DATA/vito/data/',
                        'DatasetName': 'ran_synth/img/*',
                        'fid': i,
                        'crop_size': 1024,
                        'resample_factor': 1/4,
                        'point_per_side': 30,
                        'dilation_size':15,
                        'b':100,
                        'stability_t':0.85,
                        'third_b_resample_factor':1/10, #None: use method A. 1: auto select resample rate.
                        'resolution(mm)': 0.2,
                        'expected_min_size(sqmm)': 100,
                        'min_radius': 10
                        }
                        )
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

    ar_masks=np.array(np.load(OutDIR+'Third/all_mask_third_pass_id.npy', allow_pickle=True))

    print('masks size:', ar_masks.shape)
    print(len(np.unique(ar_masks)),' mask(s) loaded')

    fn_img = glob.glob(DataDIR+DSname[:-5]+'msk/*')
    fn_img.sort()
    mask=(np.load(fn_img[fid])).astype(np.uint16)
    mask_dw=fnc.resample_fnc(mask,{'target_size':image.shape[:-1][::-1], 'method':'nearest'})
    ar_masks_rs=fnc.resample_fnc(ar_masks.astype(np.uint16),{'target_size':mask.shape[::-1], 'method':'nearest'})
    print(fn_img[fid].split("/")[-1]+' imported')
    print('No. of objects: '+str(len(np.unique(mask))-1))
    print('resampled shape: ', mask_dw.shape)

    centroids=[fnc.get_centroid(ar_masks==id) for id in np.unique(ar_masks)]
    centroids=np.array(centroids)/resample_factor

    ids, counts=np.unique(mask, return_counts=True)
    ids, counts = ids[1:], counts[1:]
    area = counts * (0.2 * 0.2)
    ids = ids[np.argsort(area)]
    area = np.sort(area)

    point_based_ac=np.zeros_like(ids)
    mask_ious=np.zeros_like(ids).astype(np.float64)
    for n in range(len(centroids))[1:]:
        hit_id=int(mask[int(centroids[n][0]),int(centroids[n][1])])
        point_based_ac[ids==hit_id]+=1
        current_iou=mask_ious[ids==hit_id]
        iou=fnc.iou(ar_masks_rs==n, mask==hit_id)
        if iou>current_iou:
            mask_ious[ids==hit_id]=iou

    np.save(DataDIR+DSname[:-5]+f'ac/point_based_ac_{i:02}.npy', [point_based_ac, mask_ious, area])