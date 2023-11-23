# Inner Speech fMRI 
Investigating inner speech using fMRI. Portfolio 4 by Mina and Anton in the course  *Advanced Cognitive Neuroscience* (E2023). 

Below, an overview of the repository is described. Please refer to the *Data Setup* and *Setup* section for instructions on how to install the necessary packages. Note that the data is highly sensitive and therefore cannot be made publicly available. Hence, the scripts are not able to run without firstly gaining access to the data. 

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
│   │   ├── ...
│   ├── button_press_sanity_check.png
│   ├── contrast_sanity_check.png
│   ├── deep_plot.png
│   ├── searchlight_deep_plot.png
│   ├── searchlight_surface_plot.png
│   ├── searchlight_topvoxels.png
│   └── surface_plot.png
├── setup.sh
└── src
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

An overview of the scripts within the `src` folder is given below: 
| Script                        | Description                                                                                      |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `first_level.py`              | Creates first-level models for all subjects. Also contains functions for loading files from BIDS (in the old data format) and getting them in the right format. |
| `sanity_check.py`             | Conducts the button press sanity check. Outputs both the plot for contrast and that for button press counts. |
| `searchlight/permutation.py`  | Performs permutation testing on the 500 most informative voxels.                                   |
| `searchlight/plot.py`         | Plots the searchlight results (surface plot & 500 most informative voxels).                        |
| `searchlight/prep.py`         | Prepares data for searchlight classification (creating first-level matrices, bmaps, conditions_label). |
| `searchlight/train.py`        | Remakes labels, reshapes data for classification, runs searchlight classification.                |
| `second_level.py`             | Creates second-level models based on first-level models from `first_level.py`. Plots whole brain contrasts and finds relevant clusters using atlas. |
| `utils.py`                    | Support functions for loading flms, removing specific subjects, and loading masks.                |

## Data Setup
Please note that you need to copy the `InSpePosNegData_copy` folder (i.e., the old BIDS file structure) to the `data` folder and rename it to `InSpePosNegData` for the code to run. Only do so on UCLOUD as the data is sensitive. 

## Setup
Before being able to run the code, please install all necessary requirements by typing in the terminal: 
```
bash setup.sh 
```
To run any code, please remember to firstly activate your virtual environment by typing `source env/bin/activate` in your terminal while being in the main folder of the directory (`cd inner-speech-fMRI`).

## Authors
This code repository was a joint effort by Anton Drasbæk Sciønning ([@drasbaek](https://github.com/drasbaek)) and Mina Almasi ([@MinaAlmasi](https://github.com/MinaAlmasi)). 