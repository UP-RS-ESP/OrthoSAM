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

para_list=[
      {'OutDIR': '/DATA/vito/output/Ravi4_run2_dw2_stb085_3b_minarea/',
      'DataDIR': '/DATA/vito/data/',
      #'DatasetName': 'sand/*',
      'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 4,
      'crop_size': 1024,
      'resample_factor': 1/2,
      'point_per_side': 30,
      'dilation_size':15,
      'b':200,
      'stability_t':0.85,
      'third_b_resample_factor':1, #None: use method A. 1: auto select resample rate.
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 100,
      'min_radius': 10
      },
      
      {'OutDIR': '/DATA/vito/output/Ravi3_run2_dw2_stb085_3b_minarea/',
      'DataDIR': '/DATA/vito/data/',
      #'DatasetName': 'sand/*',
      'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 3,
      'crop_size': 1024,
      'resample_factor': 1/2,
      'point_per_side': 30,
      'dilation_size':15,
      'b':200,
      'stability_t':0.85,
      'third_b_resample_factor':1, #None: use method A. 1: auto select resample rate.
      'resolution(mm)': 0.2,
      'expected_min_size(sqmm)': 100,
      'min_radius': 10
      }
      ]
for para in para_list:
    start_run = time.time()
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
    pre_para={'Resample': {'fxy':resample_factor},
            #'Gaussian': {'kernel size':3}
            #'CLAHE':{'clip limit':2}#,
            #'Downsample': {'fxy':4},
            #'Buffering': {'crop size': crop_size}
            }
    OutDIR=para.get('OutDIR')
    third_b=para.get('third_b_resample_factor')

    # create dir if output dir not exist
    fnc.create_dir_ifnotexist(OutDIR)

    # Save init_para to a JSON file
    with open(OutDIR+'init_para.json', 'w') as json_file:
        json.dump(para, json_file, indent=4)
    with open(OutDIR+'pre_para.json', 'w') as json_file:
        json.dump(pre_para, json_file, indent=4)

    print('Performing first pass and second pass clipwise segmentation')
    subprocess.run(["python", first_second_run, OutDIR])

    print('Merging windows')
    subprocess.run(["python", Merging_window, OutDIR])

    
    if not third_b:
        print('Searching potential missing objects and performing third pass segmentation A')
        subprocess.run(["python", Third_pass, OutDIR])
    else:
        print('Searching potential missing objects and performing third pass segmentation B')
        subprocess.run(["python", Third_pass_b, OutDIR])

    end_run = time.time()
    print('Run took: ', end_run-start_run)
