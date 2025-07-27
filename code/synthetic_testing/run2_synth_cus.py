from run_DS_cus import run_DS
import os
import sys
from ran_synth_point_ac_shadow_cus import accuracy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import notify

# Define the paths to the scripts you want to run
DSL=['ran_synth_12_1500_shadow2_0_2'
     ]

DSL=['ran_synth_12_1500_shadow2_0_5'
     ]


for DS in DSL:
    if not os.path.exists(f'/DATA/vito/output/{DS}_noshadow'):
        os.makedirs(f'/DATA/vito/output/{DS}_noshadow')
    if not os.path.exists('/DATA/vito/data/'+DS+'_noshadow'):
        os.makedirs('/DATA/vito/data/'+DS+'_noshadow')
        print('Created '+'/DATA/vito/data/'+DS+'/'+DS+'_noshadow')
    for i in [0]:#range(12):

        #run_DS(DS, i)
        accuracy(DS,i)

    print(f'{DS} segmentation completed')
    notify(DS+' segmentation completed')



notify('All task completed')
