from run_DS import run_DS
import os
import sys
#from ran_synth_point_ac import accuracy
from ran_synth_point_ac_shadow import accuracy as accuracy_shadow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import notify
import time

# Define the paths to the scripts you want to run
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
     #'ran_synth_04_3000_bw'
     #'ran_synth_01_10_cl_std_00'
     #'ran_synth_01_10_cl_std_03'
     #'ran_synth_01_10_cl_std_48'
     #,'ran_synth_01_10_cl_std_12'
     #,'ran_synth_01_10_cl_std_24'
     #,'ran_synth_02_1500_cl_std_00'
     #,'ran_synth_02_1500_cl_std_03'
     #'ran_synth_02_1500_cl_std_48'
     #,'ran_synth_02_1500_cl_std_12'
     #,'ran_synth_02_1500_cl_std_24'
     #,'ran_synth_01_1000_bw'
     #,'ran_synth_01_10_cl_std_96'
     #,'ran_synth_01_10_cl_std_128'
     #,'ran_synth_02_1500_cl_std_96'
     #,'ran_synth_02_1500_cl_std_128'
     #,'ran_synth_01_10_cl_std_192'
     #,'ran_synth_02_1500_cl_std_192'
     #'ran_synth_12_1500_shadow_0_1',
     #'ran_synth_12_1500_shadow2_0_2'#,
     'ran_synth_12_1500_shadow2_0_5'
     ]

#DSL=[#'ran_synth_12_1500_shadow_0_1',
     #'ran_synth_12_1500_shadow2_0_2'#,
     #'ran_synth_12_1500_shadow2_0_5'
     #]


for DS in DSL:
    print(f'Running {DS}...')
    if not os.path.exists(f'/DATA/vito/output/{DS}_test'):
        os.makedirs(f'/DATA/vito/output/{DS}_test')
    # if not os.path.exists('/DATA/vito/data/'+DS+'/'+DS):
    #     os.makedirs('/DATA/vito/data/'+DS+'/'+DS)
    #     print('Created '+'/DATA/vito/data/'+DS+'/'+DS)
    for i in range(1):
        start_all_script = time.time()
        run_DS(DS, i)
        end_all_script = time.time()
        print('script took: ', end_all_script-start_all_script)
        #accuracy(DS)
        #accuracy_shadow(DS,i)

notify(f'All task completed. It took {end_all_script-start_all_script}')
