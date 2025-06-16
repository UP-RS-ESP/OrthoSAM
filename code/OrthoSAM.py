import time
import logging
from Layer_0 import predict_tiles
from Merging import merge_chunks
from Layer_n import predict_tiles_n
from utility import notify
import sys
from pathlib import Path
import warnings
import os

class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_full_logging(log_file_path='output.log'):
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout) 
        ]
    )

    sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

def orthosam(para_list):
    
    # Save para to a JSON file
    warnings.simplefilter(action='ignore', category=FutureWarning)
    OutDIR=para_list[0].get('OutDIR')
    setup_full_logging(os.path.join(OutDIR,'log.txt'))


    noti=para_list[0].get('Discord_notification')
    start_run_whole = time.time()
    for n in range(len(para_list)):
        
        if n==0:
            start_run = time.time()
            predict_tiles(para_list, n)
            end_run = time.time()       

            start_run = time.time()
            merge_chunks(para_list,n)
            end_run = time.time()
            if noti:
                notify(OutDIR+' Layer 0 completed. It took '+f'{end_run-start_run}')
        else:
            start_run = time.time()
            predict_tiles_n(para_list, n)
            end_run = time.time()
            if noti:
                notify(OutDIR+f' Layer {n} completed. It took {end_run-start_run:.2f} seconds')
                
        end_run_whole = time.time()
    print(f'Run took: {((end_run_whole -start_run_whole)/60):.2f} minutes' )
    if noti:
        notify(OutDIR+' all layers completed')

def large_orthosam(OutDIR, DatasetName,fid,resolution,custom_main_para=None, custom_pass=None):
    """
    This function is used to run the OrthoSAM on large orthomosaics with a default set of parameters.
    Arguments:
    - OutDIR (str): The path of the output directory where the results will be stored.
    - DatasetName (str): The path of the dataset directory relative to the data directory.
    - fid (str or int): The filename or the index after sorting by file name.
    - resolution (float): The image resolution in mm/pixel.
    - custom_main_para (dict, optional): Custom parameters for the main configuration. Default is None.
    - custom_pass (list of dicts, optional): Custom parameters for individual layers. Default is None.
    """
    from utility import setup
    main_para={'OutDIR':OutDIR,# where output will be stored
        'DatasetName': DatasetName,
        'fid': fid,#Filename or the index after sorting by file name.
        'resolution(mm)': resolution,#image resolution in mm/pixel
        'tile_size': 1024,
        'tile_overlap':200,
        'resample_factor': 1,#'Auto': auto select resample rate.
        'input_point_per_axis': 30,
        'dilation_size':5,
        'stability_t':0.85,
        'expected_min_size(sqmm)': resolution*resolution*30,
        'min_radius': 0,
        'Discord_notification': False,# True: send discord when finished.
        'Plotting': True# True: plot the results
        }
    #specify for individual layers. e.g. different point_per_side
    passes_para_list=[
        {'resample_factor':0.5, #'Auto': auto select resample rate.
         }
        ]
    if custom_main_para:
        main_para.update(custom_main_para)
    if custom_pass:
        passes_para_list = custom_pass
    #parameters for preprocessing. If no preprocessing is needed, leave empty or remove it.
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]

    #If no preprocessing is needed, remove pre_para_list or use None.
    passes_para_list=setup(main_para, passes_para_list, pre_para_list)


    orthosam(passes_para_list)

def compact_fine_object_orthosam(OutDIR, DatasetName,fid,resolution,custom_main_para=None, custom_pass=None):
    """
    This function is used to run the OrthoSAM on images of tightly packed fine obejcts with a default set of parameters.
    Arguments:
    - OutDIR (str): The path of the output directory where the results will be stored.
    - DatasetName (str): The path of the dataset directory relative to the data directory.
    - fid (str or int): The filename or the index after sorting by file name.
    - resolution (float): The image resolution in mm/pixel.
    - custom_main_para (dict, optional): Custom parameters for the main configuration. Default is None.
    - custom_pass (list of dicts, optional): Custom parameters for individual layers. Default is None.
    """
    from utility import setup
    main_para={'OutDIR':OutDIR,# where output will be stored
        'DatasetName': DatasetName,
        'fid': fid,#Filename or the index after sorting by file name.
        'resolution(mm)': resolution,#image resolution in mm/pixel
        'tile_size': 512,
        'tile_overlap':200,
        'resample_factor': 2,#'Auto': auto select resample rate.
        'input_point_per_axis': 30,
        'dilation_size':5,
        'stability_t':0.85,
        'expected_min_size(sqmm)': resolution*resolution*30,
        'min_radius': 0,
        'Discord_notification': False,# True: send discord when finished.
        'Plotting': True# True: plot the results
        }
    #specify for individual layers. e.g. different point_per_side
    passes_para_list=[
        {'resample_factor':1, #'Auto': auto select resample rate.
         },
        {'resample_factor':0.5,
         }
        ]
    if custom_main_para:
        main_para.update(custom_main_para)
    if custom_pass:
        passes_para_list = custom_pass
    #parameters for preprocessing. If no preprocessing is needed, leave empty or remove it.
    pre_para_list=[{#'Gaussian': {'kernel size':3},
                    #'CLAHE':{'clip limit':2},
                    #'Downsample': {'fxy':4},
                    #'Buffering': {'crop size': crop_size}
                },{},{}]

    #If no preprocessing is needed, remove pre_para_list or use None.
    passes_para_list=setup(main_para, passes_para_list, pre_para_list)


    orthosam(passes_para_list)