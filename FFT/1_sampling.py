import sys
sys.path.append('../ESM')  # Path to the ESM folder
import numpy as np
import pickle
import scipy.fft as fft
from pathlib import Path
import pandas as pd
import re
import os
import torch
import matplotlib.pyplot as plt  
import random 
from itertools import product
from tqdm import tqdm
random.seed(42)
np.random.seed(42)

# qsft and ESM-specific packages
from src.utils import get_file_path, get_protein_path, load_esm_model
from fourier_utils import gwht, extract_wildtype, sampling_function, sampling


"""
Setup
"""
num_repeats = 19
q = 2
protein_bank = {}
df = pd.read_csv('positions.csv')
grouped = df.groupby('Protein')
for protein, group in grouped:
    positions = []
    for pos in group['Positions']:
        positions.append(np.fromstring(pos[1:-1], dtype=int, sep=' '))
    protein_bank[protein] = positions

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model, alphabet, batch_converter, repr_layer = load_esm_model("esm2_t36_3B_UR50D", device)

all_proteins = []
all_sparsity = []
all_ruggedness = []
all_positions = []
all_repeats = []
all_notes = []

ruggedness_all = []
ruggedness_std = []
sparsity_all = []
sparsity_std = []

labels = df['Label']

for (protein, pos), label in zip(protein_bank.items(), df['Label']):
    # wt = wild type protein
    wt_sequence = np.load(get_protein_path('wt_sequence.npy', protein))
    with open(get_file_path('aminoacid_dictionary.pkl'), 'rb') as file:
        aa_dict = pickle.load(file)
    current_directory = Path("../ESM")
    protein_path = current_directory / "fasta" / f"{protein}.fasta"
    index_to_aminoacid = {v: k for k, v in aa_dict.items()}
    wildtype = extract_wildtype(protein_path)
    

    """
    Sampling
    """
    for positions in pos: 

        # Randomize mutations
        wt = [aa_dict[wt_sequence[ind-1]] for ind in positions]
        random_matrix = np.random.choice([i for i in range(20) if i not in wt], size=(num_repeats, len(wt)), replace=True)
        mutation_dict = []
        for i in range(len(positions)):
            wt_and_mutations = np.concatenate(([wt[i]], random_matrix[:,i]))
            mutation_dict.append({index: number for index, number in enumerate(wt_and_mutations)})

        # Compute ESM Score
        n = len(positions)
        directory_name = f'results/{protein}_q{q}_n{n}_pos_{"_".join(map(str, positions))}'
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        samples = np.array(list(product(range(q), repeat=n)))  
        for i in tqdm(range(num_repeats), desc="Calculating scores"):
            file_path = f'{directory_name}/q{q}_n{n}_{i}.npy'
            if not os.path.exists(file_path):
                PLLRs = sampling(samples, positions, mutation_dict, wildtype, index_to_aminoacid, model, alphabet, batch_converter, AA_mutation=i+1, device=device)
                np.save(file_path, PLLRs)


        """
        Plotting and Analysis
        """
        ruggedness = []
        sparsity = []
        for i in range(num_repeats):

            """
            Fourier transform and coefficients
            """
            samples_ESM = np.load(f'{directory_name}/q{q}_n{n}_{i}.npy')
            F_k = np.real(gwht(samples_ESM,q,n))
            # Sort Fourier coefficients
            order = np.count_nonzero(samples, axis=1)
            sorted_indices = np.argsort(order)
            order_sorted = np.sort(order)
            F_k = F_k[sorted_indices]
            # Plot Fourier spectrum
            plt.figure()
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(F_k)
            ax.axhline(np.mean(F_k), color='C1', linestyle='--')
            plt.title(f'{protein}')
            fig.savefig(os.path.join(directory_name, f'FFT_q{q}_n{n}_{i}.png'))
            plt.close('all')
            # Cropped Fourier 
            all_std = np.std(np.abs(F_k[1:]))/5
            plt.figure()
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(F_k[1:])
            breakpoints = np.where(np.diff(order_sorted) == 1)[0] + 1
            # Plot vertical dotted lines at the breakpoints
            for breakpoint in breakpoints:
                ax.axvline(breakpoint, color='red', linestyle='--')
            last_index_order_5 = np.where(order_sorted == 5)[0][-1]
            ax.axhline(all_std, color='C2', linestyle='--')
            ax.axhline(-all_std, color='C2', linestyle='--')
            ax.set_xlim([1, last_index_order_5])
            ax.set_title(f'{protein} up to 5 interactions')
            fig.savefig(os.path.join(directory_name, f'croppedFFT_q{q}_n{n}_{i}.png'))
            plt.close('all')


            """
            Sparsity
            """
            mean_all = np.mean(np.abs(F_k[1:]))
            std_all = np.std(np.abs(F_k[1:]))
            coeffs = []
            sparsity_j = 0
            total_points = 0
            for j in range(1,np.max(order_sorted)+1):
                section_indices = order_sorted == j
                total_points_above_mean = np.sum(np.abs(F_k[section_indices]) > all_std)
                if j <= 5: # Cap at 5
                    sparsity_j += total_points_above_mean
                    total_points += len(F_k[section_indices])
                coeffs.append(total_points_above_mean/len(F_k[section_indices]))
            sparsity_norm = sparsity_j/total_points
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(np.arange(1,6), coeffs[0:5], edgecolor='black', align='edge', color='limegreen')
            ax.set_xlabel('$k^{th}$ order interactions')
            ax.set_ylabel('Fraction of nonzero coefficients')
            ax.set_xticks(np.arange(1,6))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title(f'Sparsity: {round(sparsity_norm, 3)}')
            fig.savefig(os.path.join(directory_name, f'sparsity_q{q}_n{n}_{i}.png'))
            plt.close('all')
            sparsity.append(sparsity_norm)


            """
            Ruggedness
            """
            # Percent variance explained
            var = []
            total_var = sum(F_k[1:]**2)
            for j in range(1,6):
                section_indices = order_sorted == j
                order_var = sum(F_k[section_indices]**2)
                var.append(order_var/total_var*100)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(np.arange(1,6), var, edgecolor='black', align='edge', color='limegreen')
            ax.set_xlabel('$k^{th}$ order interactions')
            ax.set_ylabel('% Variance Explained')
            ax.set_xticks(np.arange(1,6))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            weighted_avg = np.average(np.arange(1,6), weights=var)
            plt.title(f'Ruggedness: {round(weighted_avg,3)}')
            fig.savefig(os.path.join(directory_name, f'ruggedness_q{q}_n{n}_{i}.png'))
            plt.close('all')
            ruggedness.append(weighted_avg)

            all_proteins.append(protein)
            all_positions.append(np.array2string(positions))
            all_repeats.append(i)                           
            all_notes.append(label)

        ruggedness_all.append(np.mean(ruggedness))
        sparsity_all.append(np.mean(sparsity))
        ruggedness_std.append(np.std(ruggedness))
        sparsity_std.append(np.std(sparsity))

        all_sparsity = np.concatenate((all_sparsity, sparsity))
        all_ruggedness = np.concatenate((all_ruggedness, ruggedness))

data = {'Protein': all_proteins,
    'Sparsity': all_sparsity,
    'Ruggedness': all_ruggedness,
    'Positions': all_positions,
    'Experiment number': all_repeats,
    'Notes': all_notes}

df = pd.DataFrame(data)
df.to_csv(os.path.join(directory_name, '..', 'fourier_landscape_all.csv'), index=False)

protein_name = []
for protein, pos in protein_bank.items():
    protein_name = np.concatenate((protein_name, [protein] * len(protein_bank[protein])))

data = {'Protein': protein_name,
    'Sparsity': sparsity_all,
    'Sparsity std': sparsity_std,
    'Ruggedness': ruggedness_all,
    'Ruggedness std': ruggedness_std,
    'Label': labels}

df = pd.DataFrame(data)
df.to_csv(os.path.join(directory_name, '..', 'fourier_landscape.csv'), index=False)

