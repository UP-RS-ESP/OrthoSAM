import time
from OrthoSAM import orthosam
from OrthoSAM.utility import setup, notify


id_list=[#'6_7.JPG','5_6.JPG','9_5.JPG','8.JPG',
         #'11_2.JPG','13_2.JPG','152.jpg','0177_cu_2.jpg',
         #'0551_cu_1.jpg','675.jpg','1006.jpg','1013.jpg',
         #'1047.jpg',
         '1455.jpg'
         #,'2325.jpg'
         ]
for id in id_list:
    start_run = time.time()
    #Base parameters
    main_para={'OutDIR': f'SediNet_Grain_{id[:-4]}',# where output will be stored  relative to the MainOutDIR stored in config.json
        'DatasetName': "sedinet/SediNet/images",
        'fid': id,#Filename or the index after sorting by file name.
        'resolution(mm)': 1,#image resolution in mm/pixel
        'tile_size': 1024,
        'tile_overlap':200,
        'resample_factor': 1,#'Auto': auto select resample rate.
        'input_point_per_axis': 30,
        'dilation_size':5,
        'stability_t':0.85,
        'expected_min_size(sqmm)': 500,
        'min_radius': 0,
        'Calculate_stats': False, # True: calculate statistics. If you wish to use this feature, please create a file name DWH.txt in the code directory and set the webhook.
        'Discord_notification': True,# True: send discord when finished.
        'Plotting': True# True: plot the results
        }
    #specify for individual layers. e.g. different point_per_side
    passes_para_list=[
        {'resample_factor':0.5, #'Auto': auto select resample rate.
         }
        ]
    #parameters for preprocessing. If no preprocessing is needed, leave empty or remove it.
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]
    try:
        #If no preprocessing is needed, remove pre_para_list or use None.
        passes_para_list=setup(main_para, passes_para_list, pre_para_list)
        orthosam(passes_para_list)
    except Exception as e:
        print(f'Error processing {id}: {e}')

    end_run = time.time()
    print('Run took: ', end_run-start_run)
    notify(f'OrthoSAM SediNet Grain run for {id} completed in {(end_run - start_run)/60:.2f} minutes.')

# from OrthoSAM.Core import compact_fine_object_orthosam
# id_list=['4_75.JPG','393.jpg','0477.jpg','636.jpg']
# for id in id_list:
#     start_run = time.time()
#     try:
#         compact_fine_object_orthosam(f'SediNet_Sand_{id[:-4]}_compact', "sedinet/SediNet/images", id, 1)
#     except Exception as e:
#         print(f'Error processing {id}: {e}')
#     end_run = time.time()
#     print('Run took: ', end_run-start_run)
#     notify(f'OrthoSAM SediNet Sand run for {id} completed in {(end_run - start_run)/60:.2f} minutes.')

