import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import setup
from OrthoSAM import orthosam

def run_DS(DS, i):
    # Define the paths to the scripts you want to run
    master_para={'OutDIR': f'/DATA/vito/output/{DS}/{DS}_{i:02}_b250/',
        'DataDIR': '/DATA/vito/data/',
        'DatasetName': f'{DS}/img/*',
        'fid': i,
        'crop_size': 1024,
        'resample_factor': 1,
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

    OutDIR=setup(master_para, para_list, pre_para_list)
    orthosam(OutDIR)