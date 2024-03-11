# On Recovering Higher-order Interactions from Protein Language Models

This repository contains code for the paper:

*"On Recovering Higher-order Interactions from Protein Language Models"*, Darin Tsui, Amirali Aghazadeh

## Description of folders

**ESM/:** Code needed to run q-sft for ESM. More information has been left in the folder.

**FFT/:** Code needed to run Walsh-Hadamard analysis for ESM. More information has been left in the folder.

**qsft/:** Functions needed to run q-sft.

## Quick Start

### Generate all necessary files
```console
cd ESM/
mkdir model/
cd scripts/
sh 0_setup.sh
```

### Run q-sft 
```console
cd ESM/scripts/
sh 1_helper.sh
```

### Run Walsh-Hadamard Analysis 
```console
cd FFT/
python 1_sampling.py
```

Visualize reuslts using `2_analysis.ipynb`.

