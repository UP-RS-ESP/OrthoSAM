import subprocess
import json
import os
import time

start_run = time.time()

# Define the paths to the scripts you want to run
first_second_run = "code/First_second_pass.py"
Merging_window = "code/Merging_window.py"
Third_pass = 'code/Third_pass.py'

para={'OutDIR': '/DATA/vito/output/Ravi2_run_dw4/',
      'DataDIR': '/DATA/vito/data/',
      'DatasetName': 'Ravi/*',
      #'DatasetName': 'example/*',
      'fid': 0,
      'crop_size': 1024,
      'resample_factor': 1/4
      }
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

print('Performing first pass and second pass clipwise segmentation')
subprocess.run(["python", first_second_run, OutDIR])
print('Completed')

print('Merging windows')
subprocess.run(["python", Merging_window, OutDIR])
print('Completed')

print('Searching potential missing objects and performing third pass segmentation')
subprocess.run(["python", Third_pass, OutDIR])
print('Completed')

end_run = time.time()
print('Run took: ', end_run-start_run)