import subprocess
import json
import os
import time
import glob
import sys
import functions as fnc

start_run = time.time()

# Define the paths to the scripts you want to run
Merging_window = "code/Merging_window_newtile.py"
Third_pass = 'code/Third_pass_newtile.py'


OutDIR='/DATA/vito/output/Ravi2_run2_dw8_cp1024_pps48/'

print('Merging windows')
#subprocess.run(["python", Merging_window, OutDIR])

print('Searching potential missing objects and performing third pass segmentation')
subprocess.run(["python", Third_pass, OutDIR])

end_run = time.time()
print('Run took: ', end_run-start_run)