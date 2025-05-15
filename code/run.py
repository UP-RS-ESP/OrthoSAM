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
    master_para={'OutDIR': f'/DATA/vito/output/Sedinet_select/sedinet_{id}_org_dw2/',
        'DataDIR': '/DATA/vito/data/',
        'DatasetName': 'sedinet/SediNet/images/*',
        'fid': id,
        'crop_size': 1024,
        'resample_factor': 1,#None: use method A. 'Auto': auto select resample rate.
        'point_per_side': 30,
        'dilation_size':5,
        'min_size_factor':0.0001,
        'b':100,
        'stability_t':0.85,
        'resolution(mm)': 1,
        'expected_min_size(sqmm)': 500,
        'min_radius': 0
        }
    #specify for individual layers. e.g. different point_per_side
    para_list=[
        {},
        {'n_pass_resample_factor':0.5, #None: use method A. 'Auto': auto select resample rate.
        }
        ]
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]

    
    OutDIR=setup(master_para, para_list, pre_para_list)
    orthosam(OutDIR)

    end_run = time.time()
    print('Run took: ', end_run-start_run)
