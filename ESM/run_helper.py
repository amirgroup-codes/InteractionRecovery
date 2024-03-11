# Python packages
import numpy as np
import sys
from pathlib import Path
sys.path.append("..")
from src.helper import Helper
import pickle
import time
import matplotlib.pyplot as plt
import os
import pandas as pd

# q-sft-specific packages
from src.utils import get_file_path, get_protein_path

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run q-sft.")
    parser.add_argument("--q", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--noise_sd", required=True)
    parser.add_argument("--num_subsample", required=True)
    parser.add_argument("--num_repeat", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--region", required=False)
    parser.add_argument("--random", required=False, default=True)
    parser.add_argument("--hyperparam", required=False, default=False)
    return parser.parse_args()

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    
if __name__ == "__main__":
    np.random.seed(20)
    start_time = time.time()

    args = parse_args()

    """
    Setup
    """
    protein = args.protein
    # wt = wild type protein
    template_signal = np.load(get_protein_path('wt_protein.npy', protein))
    len_seq = len(template_signal)
    with open(get_file_path('aminoacid_dictionary.pkl'), 'rb') as file:
        aa_dict = pickle.load(file)
    with open(get_protein_path('WT_amino_acids.pkl', protein), 'rb') as file:
        wt_aa_dict = pickle.load(file)


    """
    q-sft parameters
    """
    # q = Number of possible amino acids
    # n = Number of sequences that are mutated
    # b = inner dimension of subsampling
    # noise_sd = hyperparameter: proxy for the amount of additive noise in the signal
    # q = 20
    # n = 10
    # b = 4
    q = int(args.q)
    n = int(args.n)
    N = q ** n
    b = int(args.b)

    # Optimal values: randomcoils = 37.5, empiricalsamples = 43.3
    # noise_sd = 37.5
    noise_sd = float(args.noise_sd)

    # num_sample and num_repeat control the amount of samples computed 
    # num_subsample = 2
    # num_repeat = 2
    num_subsample = int(args.num_subsample)
    num_repeat = int(args.num_repeat)

    # random = True: randomly select positions for sampling
    # random = False: manually input positions (1-indexed)
    # hyperparam: hyperparameter tune noise_sd
    random = str_to_bool(args.random)
    hyperparam = str_to_bool(args.hyperparam)

    # Other q-sft parameters - leave as default
    t = 3
    delays_method_source = "identity" 
    delays_method_channel = "nso"


    """
    Initialization
    """
    current_directory = Path(__file__).resolve().parent
    # Select folder
    if args.region:
        region = args.region
        folder = current_directory / "results" / f"{protein}" / f"q{q}_n{n}_b{b}_{region}"
    else:
        folder = current_directory / "results" / f"{protein}" / f"q{q}_n{n}_b{b}"
    folder.mkdir(parents=True, exist_ok=True)
    protein_path = current_directory / "fasta" / f"{protein}.fasta"

    query_args = {
        "query_method": "complex",
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "qsft",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "t": t,
        "folder": folder 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "noise_sd":noise_sd,
                    "query_args":query_args,
                    "template_signal":template_signal,
                    "len_seq":len_seq,
                    "aa_dict":aa_dict,
                    "t": t,
                    "wt_aa_dict": wt_aa_dict,
                    "aa_dict": aa_dict,
                    "protein_path": protein_path,
                    }
    test_args = {
            "n_samples": 10000
        }


    if random:
        # Random (0-indexing)
        positions = list(np.sort(np.random.choice(len_seq, size=signal_args["n"], replace=False)))
        positions = [x - 1 for x in positions]
        positions_print = [x + 1 for x in positions]
        print("Positions (1-indexing): ", positions_print)
    else:
        # Deterministic (1-indexing)
        # - Random coils: positions = [12, 15, 6, 47, 40, 9, 37, 50, 21, 38]
        # - Empirical sites: positions = [41, 39, 31, 43, 27, 30, 54, 38, 23, 33]
        positions = [12, 15, 6, 47, 40, 9, 37, 50, 21, 38]
        print("Positions (1-indexing): ", positions)
        positions = [x - 1 for x in positions]
    signal_args.update({
            "positions": positions
        })
    with open(str(folder) + "/" + "positions.pickle", "wb") as pickle_file:
        pickle.dump(signal_args["positions"], pickle_file)
    

    """
    q-sft pipeline
    """
    print('----------')
    print("Sampling from model")
    start_time_sampling = time.time()
    helper = Helper(signal_args=signal_args, methods=["qsft"], subsampling_args=query_args, test_args=test_args, exp_dir=folder)
    end_time_sampling = time.time()
    elapsed_time_sampling = end_time_sampling - start_time_sampling
    print(f"Sampling time: {elapsed_time_sampling} seconds")

    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    test_kwargs = {}
    model_kwargs["n_samples"] = num_subsample * (helper.q ** b) * num_repeat * (helper.n + 1)

    if hyperparam:
        print('Hyperparameter tuning noise_sd:')
        start_time_hyperparam = time.time()
        noise_sd = np.arange(0, 50.1, 0.1).round(2)
        nmse_entries = []

        for noise in noise_sd:
            signal_args.update({
                "noise_sd": noise
            })
            model_kwargs["noise_sd"] = noise
            model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
            test_kwargs["beta"] = model_result.get("gwht")
            nmse = helper.test_model("qsft", **test_kwargs)
            gwht = model_result.get("gwht")
            locations = model_result.get("locations")
            n_used = model_result.get("n_samples")
            avg_hamming_weight = model_result.get("avg_hamming_weight")
            max_hamming_weight = model_result.get("max_hamming_weight")
            nmse_entries.append(nmse)
            print(f"noise_sd: {noise} - NMSE: {nmse}")

        end_time_hyperparam= time.time()
        elapsed_time_hyperparam = end_time_hyperparam - start_time_hyperparam
        min_nmse_ind = nmse_entries.index(min(nmse_entries))
        min_nmse = nmse_entries[min_nmse_ind]
        print('----------')
        print(f"Hyperparameter tuning time: {elapsed_time_hyperparam} seconds")
        print(f"noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse}")

        # Recompute qsft with the best noise_sd
        signal_args.update({
            "noise_sd": noise_sd[min_nmse_ind]
        })
        model_kwargs["noise_sd"] = noise_sd[min_nmse_ind]
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse = helper.test_model("qsft", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")

        plt.figure()
        plt.title(f'q{q}_n{n}_b{b}')
        plt.plot(noise_sd, nmse_entries, marker='o', linestyle='-', color='b')
        plt.scatter(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], color='red', marker='x', label='Min NMSE')
        plt.text(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], f'noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse:.2f}', ha='right', va='top')
        plt.xlabel('noise_sd')
        plt.ylabel('NMSE')
        plt.savefig(str(folder) + '/nmse.png')  
        df = pd.DataFrame({'noise_sd': noise_sd, 'nmse': nmse_entries})
        df.to_csv(str(folder) + '/nmse.csv', index=False)

    else:
        print('Running q-sft')
        model_kwargs["noise_sd"] = noise_sd
        start_time_qsft = time.time()
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse = helper.test_model("qsft", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")
        end_time_qsft = time.time()
        elapsed_time_qsft = end_time_qsft - start_time_qsft
        print('----------')
        print(f"q-sft time: {elapsed_time_qsft} seconds")
        print(f"NMSE is {nmse}")
        
    with open(str(folder) + "/" + "qsft_transform.pickle", "wb") as pickle_file:
        pickle.dump(gwht, pickle_file)
    with open(str(folder) + "/" + "qsft_indices.pickle", "wb") as pickle_file:
        pickle.dump(locations, pickle_file)
    sorted_items = sorted(gwht.items(), key=lambda item: np.abs(item[1]), reverse=True)
    with open(str(folder) + '/sorted_qsft_transform.pickle', 'wb') as file:
        pickle.dump(dict(sorted_items), file)


    """
    Summary of results
    """
    # Plotting interaction magnitudes
    k = locations
    F_k = gwht
    nonzero_counts = {}
    for row in k:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        if num_nonzero_indices in nonzero_counts:
            nonzero_counts[num_nonzero_indices] += 1
        else:
            nonzero_counts[num_nonzero_indices] = 1
    for num_nonzero_indices, count in nonzero_counts.items():
        print(f"There are {count} {num_nonzero_indices}-order interactions.")
    # Calculate the sum of squares of F_k for each nonzero index count
    # Create dictionary keys and values
    k_values = []
    for k_i, _ in nonzero_counts.items():
        k_values.append(k_i)
    k_values = np.sort(k_values)
    # Account for subtraction of mean:
    if 0 in k_values:
        j = 0
    else:
        j = 1
    F_k_values = np.zeros(max(np.max(k_values)+1, len(k_values)))

    for row in k:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        F_k_values[num_nonzero_indices-j] += np.abs(F_k[row])
    F_k_values = np.square(F_k_values)
    sum_squares = dict(zip(k_values,F_k_values))
    index_counts = list(sum_squares.keys())
    values = list(sum_squares.values())
    plt.figure()
    plt.bar(index_counts, values, align='center', color = 'limegreen')
    plt.xlabel('$r^{th}$ order interactions')
    plt.ylabel('Magnitude of epistasis')
    plt.xticks(index_counts)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(f'q{q}_n{n}_b{b}')
    plt.savefig(str(folder) + '/magnitude_of_interactions.png')

    file_path = 'helper_results.txt'
    with open(str(folder) + "/" + str(file_path), "w") as file:
        file.write("Positions sampled = {}\n".format(signal_args.get("positions")))
        file.write("\nTotal samples = {}\n".format(n_used))
        file.write("Total sample ratio = {}\n".format(n_used / q ** n))
        file.write("NMSE = {}\n".format(nmse))
        file.write("AVG Hamming Weight of Nonzero Locations = {}\n".format(avg_hamming_weight))
        file.write("Max Hamming Weight of Nonzero Locations = {}\n".format(max_hamming_weight))
        file.write("\nFound non-zero indices QSFT: \n")
        np.savetxt(file, locations, fmt="%d")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time} seconds")
    print('----------')