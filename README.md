# Inner Speech fMRI 
Investigating inner speech using fMRI. Portfolio 4 by Mina and Anton in the course  *Advanced Cognitive Neuroscience* (E2023). 

Below, an overview of the repository is described. For reproducibility, please refer to the *Pipeline* section. Note that the data is highly sensitive and therefore cannot be made publicly available. Hence, the scripts are not able to run without firstly gaining access to the data. 

## Structure
The repository is structured as such: 
```
├── LICENSE
├── README.md
├── data
├── requirements.txt
├── results
│   ├── atlas_reader
│   │   ├── atlasreader.png
│   │   ├── atlasreader_cluster01.png
│   │   ├── atlasreader_cluster02.png
│   │   ├── atlasreader_clusters.csv
│   │   └── atlasreader_peaks.csv
│   ├── button_press_sanity_check.png
│   ├── contrast_sanity_check.png
│   ├── deep_plot.png
│   ├── searchlight_deep_plot.png
│   ├── searchlight_surface_plot.png
│   ├── searchlight_topvoxels.png
│   └── surface_plot.png
├── setup.sh
└── src
    ├── __init__.py
    ├── first_level.py
    ├── sanity_check.py
    ├── searchlight
    │   ├── permutation.py
    │   ├── plot.py
    │   ├── prep.py
    │   └── train.py
    ├── second_level.py
    └── utils.py
```

## Pipeline
Firstly, please install all necessary requirements by typing in the terminal: 
```
bash setup.sh 
```
To run any code, please remember to firstly activate your virtual environment by typing `source env/bin/activate` in your terminal while being in the main folder of the directory (`cd inner-speech-fMRI`).
