import numpy as np
import pickle
from src.utils import get_file_path, extract_wildtype
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate files needed for q-sft.")
    parser.add_argument("--fasta_path", required=True, help="Path to the fasta file of the wild-type sequence.")
    parser.add_argument("--protein_name", required=True, help="Name of the protein.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    WT_sequence = extract_wildtype(args.fasta_path)
    protein = args.protein_name
    with open(get_file_path('aminoacid_dictionary.pkl'), 'rb') as file:
        aa_dict = pickle.load(file)
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', protein)
    os.makedirs(dir, exist_ok=True)

    # Save sequence
    np.save(f'{dir}/wt_sequence.npy', np.array(list(WT_sequence)))

    # Save wild-type dictionary
    data_aa_dicts = []
    WT_protein = []
    for i in range(len(WT_sequence)):
        aa_dict_wt = {}
        WT_protein.append(aa_dict[WT_sequence[i]])
        aa_dict_wt[0] = aa_dict[WT_sequence[i]]
        # Fill in rest of amino acids
        count = 1
        for aa_key in aa_dict.keys():
            if aa_dict[aa_key] == aa_dict[WT_sequence[i]]:
                continue
            else:
                aa_dict_wt[count] = aa_dict[aa_key]
                count += 1
        data_aa_dicts.append(aa_dict_wt)
        output_file_path = f'{dir}/WT_amino_acids.pkl'
        with open(output_file_path, 'wb') as pickle_file:
            pickle.dump(data_aa_dicts, pickle_file)

        np.save(f'{dir}/wt_protein.npy', WT_protein)