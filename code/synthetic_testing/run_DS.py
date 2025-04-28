import subprocess
import json
import os
import time
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functions as fnc

DS=sys.argv[1]
i=int(sys.argv[2])

first_second_run = "code/First_second_pass_newtile.py"
Merging_window = "code/Merging_window_newtile.py"
Third_pass = 'code/Third_pass_newtile.py'
Third_pass_b = 'code/Third_pass_newtile_b.py'
noti='/DATA/vito/code/notification.py'

# Define the paths to the scripts you want to run
master_para={'OutDIR': f'/DATA/vito/output/{DS}/{DS}_{i:02}_b250/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': f'{DS}/img/*',
      'fid': i,
      'crop_size': 1024,
      'resample_factor': 1,
      '1st_resample_factor': 1,
      'point_per_side': 30,
      'dilation_size':15,
      'b':100,
      'stability_t':0.85,
      'third_b_resample_factor':1/12, #None: use method A. 1: auto select resample rate.
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 0,
      'min_radius': 0
      }
para_list=[
      {},
      {'n_pass_resample_factor':1/12, #None: use method A. 1: auto select resample rate.
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
    start_run_whole = time.time()
    if n==0:
        start_run = time.time()
        print('Performing first pass and second pass clipwise segmentation')
        subprocess.run(["python", first_second_run, OutDIR])
        end_run = time.time()
        subprocess.run(["python", noti, sys.argv[1]+' '+sys.argv[2]+' first pass completed. It took '+f'{end_run-start_run}'])


        print('Merging windows')
        start_run = time.time()
        subprocess.run(["python", Merging_window, OutDIR])
        end_run = time.time()
        subprocess.run(["python", noti, sys.argv[1]+' '+sys.argv[2]+' first pass merging completed. It took '+f'{end_run-start_run}'])
    else:
        third_b=lst[n].get('n_pass_resample_factor')
        if not third_b:
            print('Searching potential missing objects and performing third pass segmentation A')
            subprocess.run(["python", Third_pass, OutDIR])
        else:
            print('Searching potential missing objects and performing third pass segmentation B')
            start_run = time.time()
            subprocess.run(["python", Third_pass_b, OutDIR, f'{n}'])
            end_run = time.time()
        subprocess.run(["python", noti, sys.argv[1]+' '+sys.argv[2]+f' {n} pass merging completed. It took '+f'{end_run-start_run}'])

    end_run_whole = time.time()
    print('Run took: ', end_run_whole -start_run_whole )

for para in para_list:
    print(f'{para.get('OutDIR')} completed')


subprocess.run(["python", noti, sys.argv[0]])