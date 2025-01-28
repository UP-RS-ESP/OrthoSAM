import subprocess
import json
import os
import time
import glob
import sys
import functions as fnc



# Define the paths to the scripts you want to run
first_second_run = "code/First_second_pass_newtile.py"
Merging_window = "code/Merging_window_newtile.py"
Third_pass = 'code/Third_pass_newtile.py'
Third_pass_b = 'code/Third_pass_newtile_b.py'

# Define the paths to the scripts you want to run
master_para={'OutDIR': '/DATA/vito/output/Ravi4_run2_dw4_cp512_3b_minarea/',
      'DataDIR': '/DATA/vito/data/',
      #'DatasetName': 'sand/*',
      'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 4,
      'crop_size': 512,
      'resample_factor': 1/4,
      'point_per_side': 30,
      'dilation_size':15,
      'b':100,
      'stability_t':0.85,
      'n_pass_resample_factor':1/10, #None: use method A. 1: auto select resample rate.
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 100,
      'min_radius': 10
      }
para_list=[
      {'resample_factor': 1/4,
      'point_per_side': 30,
      'dilation_size':15,
      'b':100,
      'stability_t':0.85,
      'n_pass_resample_factor':1/10, #None: use method A. 1: auto select resample rate. For n pass update here only. do not touch resample factor in the master para
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 100,
      'min_radius': 10
      },
      {'resample_factor': 1/4,
      'point_per_side': 30,
      'dilation_size':15,
      'b':100,
      'stability_t':0.85,
      'n_pass_resample_factor':1, #None: use method A. 1: auto select resample rate.
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 100,
      'min_radius': 10
      },
      {'resample_factor': 1/4,
      'point_per_side': 30,
      'dilation_size':15,
      'b':200,
      'stability_t':0.85,
      'n_pass_resample_factor':1/10, #None: use method A. 1: auto select resample rate.
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 100,
      'min_radius': 10
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


for n in range(len(para_list)):
    start_run = time.time()
    if n==0:
        print('Performing first pass and second pass clipwise segmentation')
        subprocess.run(["python", first_second_run, OutDIR])

        print('Merging windows')
        subprocess.run(["python", Merging_window, OutDIR])
    else:
        third_b=lst[n].get('n_pass_resample_factor')
        if not third_b:
            print('Searching potential missing objects and performing third pass segmentation A')
            subprocess.run(["python", Third_pass, OutDIR])
        else:
            print('Searching potential missing objects and performing third pass segmentation B')
            subprocess.run(["python", Third_pass_b, OutDIR, f'{n}'])

    end_run = time.time()
    print('Run took: ', end_run-start_run)

for para in para_list:
    print(f'{para.get('OutDIR')} completed')

noti='/DATA/vito/code/notification.py'
subprocess.run(["python", noti, sys.argv[0]])