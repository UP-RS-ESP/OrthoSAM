import subprocess
import json
import os
import time
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functions as fnc
        

# Define the paths to the scripts you want to run
first_second_run = "code/First_second_pass_newtile.py"
Merging_window = "code/Merging_window_newtile.py"
Third_pass = 'code/Third_pass_newtile.py'
Third_pass_b = 'code/Third_pass_newtile_b.py'
noti='/DATA/vito/code/notification.py'
ac_py = 'code/synthetic_testing/ran_synth_point_ac.py'
run_ds='/DATA/vito/code/synthetic_testing/run_DS.py'

DSL=[#'ran_synth_08_bw'
     #,'ran_synth_16_bw'
     #,'ran_synth_32_bw'
     #,'ran_synth_64_bw'
     #,'ran_synth_64_cl_std_00'
     #,'ran_synth_64_cl_std_03'
     #,'ran_synth_64_cl_std_06'
     #,'ran_synth_64_cl_std_12'
     #,'ran_synth_64_cl_std_24'
     #,'ran_synth_08_bw_rt'
     #'ran_synth_01_10_bw'
     #,'ran_synth_04_100_bw'
     #,'ran_synth_08_100_bw'
     #,'ran_synth_01_100_bw'
     #'ran_synth_01_3000_bw'
     #,'ran_synth_02_3000_bw'
     #,'ran_synth_04_3000_bw'
     #,'ran_synth_01_10_cl_std_00'
     #'ran_synth_01_10_cl_std_03'
     #'ran_synth_01_10_cl_std_48'
     #,'ran_synth_01_10_cl_std_12'
     #,'ran_synth_01_10_cl_std_24'
     #,'ran_synth_02_1500_cl_std_00'
     #,'ran_synth_02_1500_cl_std_03'
     #,'ran_synth_02_1500_cl_std_48'
     #,'ran_synth_02_1500_cl_std_12'
     #,'ran_synth_02_1500_cl_std_24'
     #,'ran_synth_01_1000_bw'
     #,'ran_synth_01_10_cl_std_96'
     #,'ran_synth_01_10_cl_std_128'
     #,'ran_synth_02_1500_cl_std_96'
     #,'ran_synth_02_1500_cl_std_128'
     #'ran_synth_01_10_cl_std_192'
     #,'ran_synth_02_1500_cl_std_192'
     #'ran_synth_12_1500_shadow_0_1',
     'ran_synth_12_1500_shadow_0_2'
     ]


for DS in DSL:
    if not os.path.exists(f'/DATA/vito/output/{DS}'):
        os.makedirs(f'/DATA/vito/output/{DS}')
    for i in range(12):
        subprocess.run(["python", run_ds, DS, f'{i}'])

    print(f'{DS} completed')
    subprocess.run(["python", noti, DS])

    print('Working on '+DS)
    if not os.path.exists('/DATA/vito/data/'+DS+'/'+DS):
            os.makedirs('/DATA/vito/data/'+DS+'/'+DS)
            print('Created '+'/DATA/vito/data/'+DS+'/'+DS)
    print(DS)
    subprocess.run(["python", ac_py, DS])
    subprocess.run(["python", noti, DS])


subprocess.run(["python", noti, sys.argv[0]])
