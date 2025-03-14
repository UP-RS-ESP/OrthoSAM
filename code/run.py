import subprocess
import json
import os
import time
import glob
import sys
import functions as fnc

# Define the paths to the scripts you want to run
run_DS = "code/run_DS.py"
for id in range(3)[2:]:
    start_run = time.time()
    master_para={'OutDIR': f'/DATA/vito/output/Arg_clip{id}_run_2_dw1_2_b300_48pps/',
        'DataDIR': '/DATA/vito/data/',
        'DatasetName': 'Argentina/*',
        'fid': id,
        'crop_size': 1024,
        'resample_factor': 2,
        '1st_resample_factor': 2,
        'point_per_side': 48,
        'dilation_size':1,
        'min_size_factor':0.0001,
        'b':300,
        'stability_t':0.85,
        'resolution(mm)': 0.5,
        'expected_min_size(sqmm)': 10,
        'min_radius': 0
        }
    para_list=[
        {},
        {'n_pass_resample_factor':1, #None: use method A. 1: auto select resample rate.
        },
        {'n_pass_resample_factor':1/2, #None: use method A. 1: auto select resample rate.
        }
        ]
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]

    # create dir if output dir not exist
    OutDIR=master_para.get('OutDIR')
    fnc.create_dir_ifnotexist(OutDIR)

    if master_para.get('fid')==None:
        if not os.path.exists(master_para.get('DataDIR')+master_para.get('DatasetName')[:-1]):
            print('Input directory does not exist. Exiting script.')
            sys.exit()
        fn_img = glob.glob(master_para.get('DataDIR')+master_para.get('DatasetName'))
        fn_img.sort()
        for i,fn in enumerate(fn_img):
            print(i, ': ', fn)
        print('--------------')
        while True:
            try:
                user_input = int(input("Please select an image: "))
                print(f"{fn_img[user_input]} selected")
                master_para.update({'fid':user_input})
                break  # Exit the loop if the input is valid
            except ValueError:
                print("Requires an index. Please try again.")

    # Save init_para to a JSON file
    lst = [dict(master_para, **para) for para in para_list]
    with open(OutDIR+f'init_para.json', 'w') as json_file:
        json.dump(lst, json_file, indent=4)
    with open(OutDIR+f'pre_para.json', 'w') as json_file:
        json.dump(pre_para_list, json_file, indent=4)

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
    subprocess.run(["python", noti, sys.argv[0]])