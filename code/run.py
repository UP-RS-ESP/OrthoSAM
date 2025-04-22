import subprocess
import json
import os
import time
import glob
import sys
import functions as fnc

run_DS = "code/run_DS.py"
#for id in [0,4,7]:

import numpy as np
id_list=np.array(range(400))
id_list=id_list[id_list!=382]
for id in id_list:

    id=int(id)

    start_run = time.time()
    master_para={'OutDIR': f'/DATA/vito/output/Sedinet/sedinet_{id}_up2_org/',
        'DataDIR': '/DATA/vito/data/',
        'DatasetName': 'sedinet/SediNet/images/*',
        'fid': id,
        'crop_size': 1024,
        'resample_factor': 1,
        '1st_resample_factor': 1,
        'point_per_side': 48,
        'dilation_size':1,
        'min_size_factor':0.0001,
        'b':300,
        'stability_t':0.85,
        'resolution(mm)': 0.39,
        'expected_min_size(sqmm)': 0,
        'min_radius': 0
        }
    para_list=[
        {},
        {'n_pass_resample_factor':0.5, #None: use method A. 1: auto select resample rate.
        }
        ]
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]

    if not os.path.exists(master_para.get('DataDIR')+master_para.get('DatasetName')[:-1]):
        print('Input directory does not exist. Exiting script.')
        sys.exit()

    # create dir if output dir not exist
    OutDIR=master_para.get('OutDIR')
    fnc.create_dir_ifnotexist(OutDIR)
    if master_para.get('fid')==None:
        master_para=fnc.prompt_fid(master_para)


    # Save init_para to a JSON file
    lst = [dict(master_para, **para) for para in para_list]
    with open(OutDIR+f'init_para.json', 'w') as json_file:
        json.dump(lst, json_file, indent=4)
    with open(OutDIR+f'pre_para.json', 'w') as json_file:
        json.dump(pre_para_list, json_file, indent=4)

    subprocess.run(["python", run_DS, OutDIR])

    end_run = time.time()
    print('Run took: ', end_run-start_run)

    noti='/DATA/vito/code/notification.py'
    subprocess.run(["python", noti, f"{sys.argv[0]} has completed successfully! It took {end_run-start_run}"])