# Harnessing Chemical Intuition from Collective Intelligence

This repository contains the associated code and datasets for the paper _Harnessing Medicinal Chemistry Intuition from Collective Intelligence_. It includes scripts for data visualization, model training/testing, and data preparation, along with pre-trained models for predicting ADMET properties.

## Installation

### Setting Up the Conda Environment

To set up the required environment, you first need to clone this repository to your local machine:

```bash
git clone https://github.com/Sanofi-GitHub/IDD-Collective-Intelligence.git
cd IDD-Collective-Intelligence
```

Then, you can create a conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
conda activate Chemical_Intuition_from_Collective_Intelligence
```

This will install all the necessary packages and dependencies required to run the scripts in this repository.

### Repository Structure

- 0_Data_Vizualisation.ipynb: Jupyter notebook for data visualization.
- 1_TrainTest_GNN.ipynb: Jupyter notebook for training and testing graph neural networks.
- 2_AllCombinations_CI.ipynb: Jupyter notebook for testing all combinations of collective intelligence inputs.
- _all_functions.py: Python script containing all custom functions used across notebooks.
- data/: Directory containing datasets for ADMET properties.
- figure/: Directory containing figures generated from the notebooks.
- model/: Directory containing pre-trained models for ADMET properties.
- environment.yml: Conda environment file.


### Additional Notes
Figures are stored under the figure/ directory in both PNG and SVG formats.
Data used in the notebooks are located in the data/ directory, structured by ADMET property.
Pre-trained models can be found in the model/ directory, categorized by the property they predict.

### Usage
This work make available the human response from the `Collective Intelligence` exercise. 
We provide and use by default a pre-trained predictive models trained on LogP, LogD, Water solubility, Apparent permeability, and hERG inhibition. 
Training data are additionally provided.

After setting up your environment, you can start by running the Jupyter notebooks.
A CUDA-enabled GPU is not required for usage, but recommended for speed. 

### Credits and contact
This material has been prepared by Pierre Llompart (Sanofi R&D and University of Strasbourg, France) and supervised by Paraskevi Gkeka (Sanofi R&D, France). For any technical question, please write to Pierre.Llompart@sanofi.com or Paraskevi.Gkeka@sanofi.com.