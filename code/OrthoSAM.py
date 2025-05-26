import time
import logging
from Layer_0 import predict_tiles
from Merging import merge_chunks
from Layer_n import predict_tiles_n
from utility import notify
import sys
from pathlib import Path
import warnings

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message.rstrip() != '':
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def setup_full_logging(log_file_path='output.log'):
    # Create log directory if needed
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)  # still show in terminal
        ]
    )

    # Redirect print(), warnings, errors
    sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

def orthosam(para_list):
    #setup_full_logging('logs/output.log')
    # Save para to a JSON file
    warnings.simplefilter(action='ignore', category=FutureWarning)
    OutDIR=para_list[0].get('OutDIR')
    #with open(os.path.join(OutDIR,'para.json'), 'r') as json_file:
    #    para_list = json.load(json_file)

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