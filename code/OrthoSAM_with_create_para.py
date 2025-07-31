import time
from OrthoSAM import orthosam
from utility import setup

#for id in [0,4,7]:

#import numpy as np
#id_list=np.array(range(400))
#id_list=id_list[id_list!=382]
id_list=["13_2.JPG","8.JPG","6_7.JPG","9_5.JPG","5_6.JPG"]
for id in id_list:

    #id=int(id)

    start_run = time.time()
    #Base parameters
    main_para={'OutDIR': f'Sedinet_select/sedinet_{id}_org_dw2',# where output will be stored  relative to the MainOutDIR stored in config.json
        'DatasetName': 'sedinet/SediNet/images',
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

    #If no preprocessing is needed, remove pre_para_list or use None.
    passes_para_list=setup(main_para, passes_para_list, pre_para_list)


    orthosam(passes_para_list)

    end_run = time.time()
    print('Run took: ', end_run-start_run)
