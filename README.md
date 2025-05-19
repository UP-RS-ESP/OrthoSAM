# OrthoSAM: Extending the Capabilities of SegmentAnything to Delineate River Pebbles from Orthophotos


Sediment characteristics and grain-size distribution are crucial for understanding natural hazards, hydrologic conditions, and ecosystems. However, traditional methods for collecting this information are costly, labor-intensive, and time-consuming. To address this, we present OrthoSAM, a workflow leveraging the Segment Anything Model (SAM) for automated delineation of densely packed pebbles in high-resolution orthomosaics. Our framework consists of a tiling scheme, improved seed (input) point generation, and a multi-scale resampling scheme. Validation using synthetic images shows high precision close to 1, a recall above 0.9, with a mean IoU above 0.9. Using a large synthetic dataset, we show that the two-sample Kolmogorov-Smirnov test confirms the accuracy of the grain size distribution. We identified a size detection limit of 700 pixels and a noise limit at $\sigma$ = 96. Applying OrthoSAM to orthomosaics from the Ravi River in India, we delineated 6087 pebbles with high precision and recall. The resulting grain dataset includes measurements such as area, axis lengths, perimeter, RGB statistics, and smoothness, providing valuable insights for further analysis in geomorphology and ecosystem studies.

**Workflow**
![Pebble Flow Chart](fig/pebble_flow_chart.png)

## Dependencies
* Python 3.11+

For required packages, please see [requirements.txt](requirements.txt). This project was developed and tested using Python's built-in virtual environment module, `venv`.

Additionally, the code requires `python>=3.8`, `pytorch>=1.7` and `torchvision>=0.8`. The installation instructions can be found [here](https://pytorch.org/get-started/locally/).

## Setup guide
### Installation with conda:

1. Install environment:
```bash
conda create -n OrthoSAM -c conda-forge python=3.11 pip ipython jupyterlab numpy pandas numba scipy scikit-learn scikit-image matplotlib cupy pytorch torchvision
conda activate OrthoSAM
conda install -c pytorch torch 
```

2. Install requirements: 
```bash
conda activate OrthoSAM
pip install -r requirements.txt
```

3. Install conda kernel for jupyter lab:
```bash
cd OrthoSAM/code
python -m ipykernel install --user --name=OrthoSAM
```

4. Create a subdirectory for storing model checkpoints and download SAM checkpoints:
```bash
mkdir -p data/MetaSAM
cd data/MetaSAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

```


### Installation with a virtual environment
1. Create a virtual environment
```
python -m venv venv
```


2. Activate the virtual environment

On macOS/Linux:
```
source venv/bin/activate
```
On Windows:
```
venv\Scripts\activate
```

3. To install all required packages: 
```
pip install -r requirements.txt
```

4. Create a subdirectory for storing model checkpoints
```bash
mkdir -p MetaSAM
```

5. Download the Segment Anything checkpoint from 


`vit_h`:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

`vit_l`:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

`vit_b`:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Download all three:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

6. Move the downloaded checkpoint into the MetaSAM folder

7. Update configuration path. Please update the data directory and checkpoint directory path in [`config.json`](code/config.json). This can be done automatically with [`update_config.py`](code/update_config.py). 
    - This is also the file to specify which checkpoint to use. If you wish set any default parameter, it can be added to `config.json`. Please note that parameters defined in the script has the priority.
## Codes
- [`OrthoSAM_notebook.ipynb`](code/OrthoSAM_notebook.ipynb): Instruction of how to create parameters and run OrthoSAM.
- [`para_helper.ipynb`](code/para_helper.ipynb): Parameter assistance.
- [`OrthoSAM_with_create_para.py`](code/OrthoSAM_with_create_para.py): Script to create parameters and run OrthoSAM.
- [`update_config.py`](code/update_config.py): Update data directory and checkpoint directory path.
- [`config.json`](code/config.json): Configuration file to define model type, checkpoint directory, data directory, and any default parameters.
<!-- `OrthoSAM`: [OrthoSAM codes.](code/OrthoSAM.py)-->



## Segment-Anything

We have developed our framework based on the Meta AI Segment-Anything model. For more details regarding the model please visit their Github:
https://github.com/facebookresearch/segment-anything
