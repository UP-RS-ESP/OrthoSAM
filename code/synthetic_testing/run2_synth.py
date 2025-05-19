from run_DS import run_DS
import os
import sys
from ran_synth_point_ac_shadow import accuracy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from notification import notify

# Define the paths to the scripts you want to run
ac_py = 'code/synthetic_testing/ran_synth_point_ac_shadow.py'


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
     #'ran_synth_12_1500_shadow2_0_2',
     'ran_synth_12_1500_shadow2_0_5'
     ]


for DS in DSL:
    if not os.path.exists(f'/DATA/vito/output/{DS}'):
        os.makedirs(f'/DATA/vito/output/{DS}')
    if not os.path.exists('/DATA/vito/data/'+DS+'/'+DS):
        os.makedirs('/DATA/vito/data/'+DS+'/'+DS)
        print('Created '+'/DATA/vito/data/'+DS+'/'+DS)
    for i in range(12):
        run_DS(DS, i)
        accuracy(DS,i)

    print(f'{DS} segmentation completed')
    notify(DS+' segmentation completed')


notify('All task completed')
