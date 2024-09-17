import subprocess
import json
import os
import time
import glob
import sys

start_run = time.time()

# Define the paths to the scripts you want to run
first_second_run = "code/First_second_pass.py"
Merging_window = "code/Merging_window.py"
Third_pass = 'code/Third_pass.py'

para={'OutDIR': '/DATA/vito/output/Sand2_4096_run_dw1_cp512_pps80/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'sand/*',
      #'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 3,
      'crop_size': 512,
      'resample_factor': 1,
      'point_per_side': 80,
      'dilation_size':15,
      'min_size_factor':0.0001,
      'window_step':0.5
      }

if para.get('fid')==None:
    if not os.path.exists(para.get('DataDIR')+para.get('DatasetName')[:-1]):
        print('Input directory does not exist. Exiting script.')
        sys.exit()
    fn_img = glob.glob(para.get('DataDIR')+para.get('DatasetName'))
    fn_img.sort()
    for i,fn in enumerate(fn_img):
        print(i, ': ', fn)
    print('--------------')
    while True:
        try:
            user_input = int(input("Please select an image: "))
            print(f"{fn_img[user_input]} selected")
            para.update({'fid':user_input})
            break  # Exit the loop if the input is valid
        except ValueError:
            print("Requires an index. Please try again.")
resample_factor=para.get('resample_factor')
pre_para={'Downsample': {'fxy':resample_factor},
        #'Gaussian': {'kernel size':3}
        #'CLAHE':{'clip limit':2}#,
        #'Downsample': {'fxy':4},
        #'Buffering': {'crop size': crop_size}
        }
OutDIR=para.get('OutDIR')

# create dir if output dir not exist
if not os.path.exists(OutDIR[:-1]):
    os.makedirs(OutDIR[:-1])
if not os.path.exists(OutDIR+'chunks'):
    os.makedirs(OutDIR+'chunks')
if not os.path.exists(OutDIR+'Merged'):
    os.makedirs(OutDIR+'Merged')
if not os.path.exists(OutDIR+'Third'):
    os.makedirs(OutDIR+'Third')

# Save init_para to a JSON file
with open(OutDIR+'init_para.json', 'w') as json_file:
    json.dump(para, json_file, indent=4)
with open(OutDIR+'pre_para.json', 'w') as json_file:
    json.dump(pre_para, json_file, indent=4)

print('Performing first pass and second pass clipwise segmentation')
#subprocess.run(["python", first_second_run, OutDIR])

print('Merging windows')
subprocess.run(["python", Merging_window, OutDIR])

print('Searching potential missing objects and performing third pass segmentation')
subprocess.run(["python", Third_pass, OutDIR])

end_run = time.time()
print('Run took: ', end_run-start_run)

para={'OutDIR': '/DATA/vito/output/Ravi3_run_dw4_cp512_pps48/',
      'DataDIR': '/DATA/vito/data/',
      #'DatasetName': 'sand/*',
      'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 3,
      'crop_size': 512,
      'resample_factor': 1/4,
      'point_per_side': 48,
      'dilation_size':15,
      'min_size_factor':0.0001,
      'window_step':0.3
      }
