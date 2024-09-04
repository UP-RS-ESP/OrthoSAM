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

para={'OutDIR': '/DATA/vito/output/Sand2_4096_run_dw1_cp512/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'sand/*',
      #'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 3,
      'crop_size': 512,
      'resample_factor': 1
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
# Save init_para to a JSON file
with open(OutDIR+'init_para.json', 'w') as json_file:
    json.dump(para, json_file, indent=4)
with open(OutDIR+'pre_para.json', 'w') as json_file:
    json.dump(pre_para, json_file, indent=4)

def run_script(cmd):
    try:
        # Run the script using subprocess
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{cmd[1]} completed successfully.")
        return result.stdout  # Return the output of the script
    except subprocess.CalledProcessError as e:
        # This block runs if the script returns a non-zero exit status
        print(f"{cmd[1]} failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        sys.exit(1)  # Exit the parent script if a child script fails

print('Performing first pass and second pass clipwise segmentation')
#subprocess.run(["python", first_second_run, OutDIR])
#run_script(["python", first_second_run, OutDIR])

print('Merging windows')
#subprocess.run(["python", Merging_window, OutDIR])
#run_script(["python", Merging_window, OutDIR])

print('Searching potential missing objects and performing third pass segmentation')
subprocess.run(["python", Third_pass, OutDIR])
#run_script(["python", Third_pass, OutDIR])

end_run = time.time()
print('Run took: ', end_run-start_run)