import time
from OrthoSAM.Core import large_orthosam, compact_fine_object_orthosam
from OrthoSAM.utility import notify

#for id in [0,4,7]:

#import numpy as np
#id_list=np.array(range(400))
#id_list=id_list[id_list!=382]
id_list=['K1.jpg','S1.jpg','FH.jpg']
for id in id_list:

    #id=int(id)

    start_run = time.time()
    #Base parameters
    DS_name='ImageGrains_v1_data/field_examples'

    large_orthosam(f'ImageGrain_{id[:-4]}_large', DS_name, id, 1)
    compact_fine_object_orthosam(f'ImageGrain_{id[:-4]}_compact', DS_name, id, 1)

    end_run = time.time()
    print('Run took: ', end_run-start_run)
    notify(f'OrthoSAM ImageGrain run for {id} completed in {(end_run - start_run)/60:.2f} minutes.')
