import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import setup
from OrthoSAM import orthosam

def run_DS(DS, i):
    # Define the paths to the scripts you want to run
    master_para={'OutDIR': f'/DATA/vito/output/{DS}/{DS}_{i:02}_b250',
        'DatasetName': f'{DS}/img',
        'fid': i,
        'resolution(mm)': 1,#image resolution in mm/pixel
        'tile_size': 1024,
        'tile_overlap':200,
        'resample_factor': 1,#'Auto': auto select resample rate.
        'input_point_per_axis': 30,
        'dilation_size':15,
        'stability_t':0.85,
        'expected_min_size(sqmm)': 0,
        'min_radius': 0,
        'Discord_notification': True,# True: send discord when finished.
        'Plotting': True#
        }
    para_list=[
        {'resample_factor':1/12, #None: use method A. 1: auto select resample rate.
        }
        ]
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]

    passes_para_list=setup(master_para, para_list, pre_para_list)
    orthosam(passes_para_list)