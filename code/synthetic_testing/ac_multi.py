import os
import subprocess
import sys

ac_py = 'code/synthetic_testing/ran_synth_point_ac.py'
DSL=['ran_synth_08_bw'
     ,'ran_synth_16_bw'
     #,'ran_synth_32_bw'
     #,'ran_synth_64_bw'
     #,'ran_synth_64_cl_std_00'
     #,'ran_synth_64_cl_std_03'
     ,'ran_synth_64_cl_std_06'
     ,'ran_synth_64_cl_std_12'
     ,'ran_synth_64_cl_std_24'
     ,'ran_synth_08_bw_rt'
     ]
for pth in DSL:
        print('Working on '+pth)
        if not os.path.exists('/DATA/vito/data/'+pth+'/'+pth):
              os.makedirs('/DATA/vito/data/'+pth+'/'+pth)
              print('Created '+'/DATA/vito/data/'+pth+'/'+pth)
        print(pth)
        subprocess.run(["python", ac_py, pth])

noti='/DATA/vito/code/notification.py'
subprocess.run(["python", noti, sys.argv[0]])