# Python standard libraries
import numpy as np
import torch
from torch.multiprocessing import Pool
import time
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import re
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os 
import sys
sys.path.append('..')
import multiprocessing as mp
from functools import partial
import uuid
from multiprocessing import Process, Queue, cpu_count

# qsft-specific packages
from qsft.utils import igwht_tensored, random_signal_strength_model, qary_vec_to_dec, sort_qary_vecs
from qsft.input_signal import Signal
from qsft.input_signal_subsampled import SubsampledSignal
from qsft.utils import dec_to_qary_vec

# ESM functions obtained from https://github.com/ntranoslab/esm-variants
from src.utils import load_esm_model, get_PLLR_batch, get_PLLR
GLOBAL_NUM_CLASSES = 20
GLOBAL_STR_ALPHABET = ""
GLOBAL_DICT_ALPHABET = {i: GLOBAL_STR_ALPHABET[i] for  i in range(len(GLOBAL_STR_ALPHABET))}
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def extract_wildtype(file_path):
    with open(file_path, 'r') as file:
        fasta_content = file.read()
    match = re.search(r'\n([A-Za-z]+)', fasta_content)
    if match:
        sequence = match.group(1)
        return sequence
    else:
        return None

def worker_process(query_indices_batch, q, sampling_function_args):
    partial_sampling_function = partial(sampling_function, **sampling_function_args)
    results = partial_sampling_function(query_indices_batch)
    q.put(results)

def sampling_function(query_batch, q, n, wt_aa_dict, wildtype, positions, index_to_aminoacid, model, alphabet, batch_converter):
    """Sampling Function."""
    random_samples = np.array(dec_to_qary_vec(query_batch, q, n)).T
    if len(random_samples.shape) > 2:
        random_samples = random_samples.squeeze(1)

    # Create random samples
    for i, ind in enumerate(positions):
        wt_dict = wt_aa_dict[ind] 
        random_samples[:, i] = np.vectorize(wt_dict.get)(random_samples[:, i])

    # Create mutations from sampling
    mutated_sequences = []
    starting_pos = []
    wt_seq = []
    for sample in random_samples:
        sequence = list(wildtype)
        wt_seq.append(wildtype)
        starting_pos.append(positions[0])
        for i, mutation in enumerate(sample):
            sequence[positions[i]] = index_to_aminoacid[mutation]
        mutated_sequence = ''.join(sequence)
        mutated_sequences.append(mutated_sequence)
    
    input_df = pd.DataFrame({'wt_seq': wt_seq, 'mut_seq': mutated_sequences, 'start_pos': starting_pos})
    
    batch_size = 512
    wt_seqs = input_df['wt_seq'].tolist()
    mut_seqs = input_df['mut_seq'].tolist()
    start_positions = input_df['start_pos'].astype(int).tolist()

    PLLRs = []
    for batch_start in tqdm(range(0, len(wt_seqs), batch_size), desc="Calculating PLLRs"):
        batch_end = batch_start + batch_size
        wt_batch = wt_seqs[batch_start:batch_end]
        mut_batch = mut_seqs[batch_start:batch_end]
        start_pos_batch = start_positions[batch_start:batch_end]
        PLLR_batch = get_PLLR_batch(wt_batch, mut_batch, start_pos_batch, model, alphabet, batch_converter, device=device)
        PLLRs.extend(PLLR_batch)
    return np.array(PLLRs)


class BioSubsampledSignal(SubsampledSignal):
    """
    This is a Subsampled signal object, except it implements the unimplemented 'subsample' function.
    """
    def __init__(self, **kwargs):
        self.q = kwargs["q"]
        self.n = kwargs["n"]
        self.noise_sd = kwargs["noise_sd"]
        self.template_signal = kwargs["template_signal"]
        self.len_seq = kwargs["len_seq"]
        self.positions = kwargs["positions"]
        self.wt_aa_dict = kwargs["wt_aa_dict"]
        self.index_to_aminoacid = {v: k for k, v in kwargs["aa_dict"].items()}

        # ESM-specific initalization 
        self.protein_path = kwargs["protein_path"]
        self.model, self.alphabet, self.batch_converter, self.repr_layer = load_esm_model("esm2_t36_3B_UR50D", device)
        self.wildtype = extract_wildtype(kwargs["protein_path"])
        self.fasta_dir = os.path.dirname(self.protein_path)

        super().__init__(**kwargs)

    def to_numpy(self, tensor):
        """
        Converts torch tensor to numpy to make compatible with onnxruntime.
        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def subsample(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        batch_size = 10000
        res = []
        query_indices_batches = np.array_split(query_indices, len(query_indices)//batch_size + 1)

        partial_sampling_function = partial(
            sampling_function,
            q=self.q,
            n=self.n,
            wt_aa_dict=self.wt_aa_dict,  
            wildtype=self.wildtype,    
            positions=self.positions,
            index_to_aminoacid=self.index_to_aminoacid,
            model=self.model,
            alphabet=self.alphabet,
            batch_converter=self.batch_converter
        )

        for query_indices_batch in query_indices_batches:
            new_res = partial_sampling_function(query_indices_batch)
            res.append(new_res)
        return np.concatenate(res)