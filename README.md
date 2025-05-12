# OrthoSAM: Extending the Capabilities of SegmentAnything to Delineate River Pebbles from Orthophotos


Sediment characteristics and grain-size distribution are crucial for understanding natural hazards, hydrologic conditions, and ecosystems. However, traditional methods for collecting this information are costly, labor-intensive, and time-consuming. To address this, we present OrthoSAM, a workflow leveraging the Segment Anything Model (SAM) for automated delineation of densely packed pebbles in high-resolution orthomosaics. Our framework consists of a tiling scheme, improved seed (input) point generation, and a multi-scale resampling scheme. Validation using synthetic images shows high precision close to 1, a recall above 0.9, with a mean IoU above 0.9. Using a large synthetic dataset, we show that the two-sample Kolmogorov-Smirnov test confirms the accuracy of the grain size distribution. We identified a size detection limit of 700 pixels and a noise limit at $\sigma$ = 96. Applying OrthoSAM to orthomosaics from the Ravi River in India, we delineated 6087 pebbles with high precision and recall. The resulting grain dataset includes measurements such as area, axis lengths, perimeter, RGB statistics, and smoothness, providing valuable insights for further analysis in geomorphology and ecosystem studies.

**Workflow**
![Pebble Flow Chart](fig/pebble_flow_chart.png)

## Dependencies
* Python 3.11+
## Setup guide
For required packages, please see [requirements.txt](requirements.txt). This project is developed and tested using Python's built-in virtual environment module, `venv`.

Additionally, the code requires `python>=3.8`, `pytorch>=1.7` and `torchvision>=0.8`. The installation instructions can be found [here](https://pytorch.org/get-started/locally/).

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
mkdir -p data/MetaSAM

5. Download the Segment Anything checkpoint from 



`vit_h`:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

`vit_l`:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

`vit_b`:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


Move the downloaded checkpoint into the MetaSAM folder

## Codes
- `OrthoSAM.ipynb`: [Instruction Example notebook.](code/OrthoSAM.ipynb)


## Segment-Anything

We developed our framework based on the Meta AI Segment-Anything model. For more details regarding the model please visit their Github:
https://github.com/facebookresearch/segment-anything