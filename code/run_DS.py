import subprocess
import json
import time
import sys
import functions as fnc

OutDIR=sys.argv[1]

# Define the paths to the scripts you want to run
first_second_run = "code/First_second_pass_newtile.py"
Merging_window = "code/Merging_window_newtile.py"
Third_pass = 'code/Third_pass_newtile.py'
Third_pass_b = 'code/Third_pass_newtile_b.py'

# Save init_para to a JSON file
with open(OutDIR+'init_para.json', 'r') as json_file:
    para_list = json.load(json_file)


for n in range(len(para_list)):
    start_run = time.time()
    if n==0:
        print('Performing first pass and second pass clipwise segmentation')
        subprocess.run(["python", first_second_run, OutDIR])

        print('Merging windows')
        subprocess.run(["python", Merging_window, OutDIR])
    else:
        third_b=para_list[n].get('n_pass_resample_factor')
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

noti='code/notification.py'
subprocess.run(["python", noti, sys.argv[0]])