# Walsh-Hadamard Analysis of ESM

If not already present, create the folder `model/` folder in `ESM`. This is where the ESM model will install to.

All files needed to run scripts have been generated and left in the `ESM` folder for GB1, GFP, and TP53. To run on other proteins, add relevant fasta files to `ESM/fasta`, and run `0_setup.sh` on those proteins located in `ESM/scripts`. 

## Running scripts

From this directory, run the following command:

```console
python 1_sampling.py
```

Visual analysis can be located in `2_analysis.ipynb`. 

## Description of files

**results/:** contains all Walsh-Hadamard results, organized by protein and positions sampled. In each folder, you can find the ESM scores sampled (`q{q}_n{n}_{run}`), plots corresponding to the Fourier transform, and plots corresponding to sparsity and ruggedness.

Results are generated using code from Brandes et al.:

```console
@article{Brandes23,
author = {Brandes, Nadav and Goldman, Grant and Wang, Charlotte H. and Ye, Chun Jimmie and Ntranos, Vasilis},
title = {Genome-wide prediction of disease variant effects with a deep protein language model},
journal = {Nature Genetics},
volume = {55},
pages = {1512--1522},
year = {2023}
}
```

- **fourier_landscape_all.csv:** csv of all experiments run for all runs

- **fourier_landscape.csv:** summary statistics for positions sampled

- **Meier_etal/:** Walsh-Hadamard results for ESM scores computed using code from Meier et al.:

```console
@inproceedings{Meier2021,
 author = {Meier, Joshua and Rao, Roshan and Verkuil, Robert and Liu, Jason and Sercu, Tom and Rives, Alex},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Language models enable zero-shot prediction of the effects of mutations on protein function},
 year = {2021}
}
```

**1_sampling.py:** script to compute the Walsh-Hadamard transform over all positions in `positions.csv`. 

- Note: To switch from Brandes et al. to Meier et al.'s ESM score, change line 87 from `sampling_function` to `sampling`.

**2_analysis.ipynb:** notebook to provide useful visuals for generating ruggedness-sparsity plots



