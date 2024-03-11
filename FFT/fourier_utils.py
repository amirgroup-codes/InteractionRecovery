import pandas as pd
import numpy as np
import scipy.fft as fft
from tqdm import tqdm
import re
import sys
sys.path.append('../ESM')  # Path to the ESM folder
from src.utils import get_PLLR_batch, get_PLLR
import torch
from torch.utils.data import Dataset, DataLoader

def gwht(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.fftn(x_tensor) / (q ** n)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def extract_wildtype(file_path):
    with open(file_path, 'r') as file:
        fasta_content = file.read()
    match = re.search(r'\n([A-Za-z]+)', fasta_content)
    if match:
        sequence = match.group(1)
        return sequence
    else:
        return None

def sampling_function(samples, positions, mutation_dict, wildtype, index_to_aminoacid, model, alphabet, batch_converter, AA_mutation=1, device=0):
    """Sampling Function."""
    # Convert 1-indexing to 0-indexing
    positions = [x - 1 for x in positions]
    samples = samples.copy()
    # Mutate according to AA_mutation
    samples[samples == 1] = AA_mutation
    for i in range(len(positions)):
        wt_dict = mutation_dict[i]
        samples[:, i] = np.vectorize(wt_dict.get)(samples[:, i])

    # Create mutations from sampling
    mutated_sequences = []
    starting_pos = []
    wt_seq = []
    for sample in samples:
        sequence = list(wildtype)
        wt_seq.append(wildtype)
        starting_pos.append(positions[0])
        for i, mutation in enumerate(sample):
            sequence[positions[i]] = index_to_aminoacid[mutation]
        mutated_sequence = ''.join(sequence)
        mutated_sequences.append(mutated_sequence)
    
    input_df = pd.DataFrame({'wt_seq': wt_seq, 'mut_seq': mutated_sequences, 'start_pos': starting_pos})
    print(input_df)
    # print(mutated_sequences)
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

def sampling(samples, positions, mutation_dict, wildtype, index_to_aminoacid, model, alphabet, batch_converter, AA_mutation=1, device=0):
    """Sampling Function."""
    # Convert 1-indexing to 0-indexing
    positions = [x - 1 for x in positions]
    samples = samples.copy()
    # Mutate according to AA_mutation
    samples[samples == 1] = AA_mutation
    for i in range(len(positions)):
        wt_dict = mutation_dict[i]
        samples[:, i] = np.vectorize(wt_dict.get)(samples[:, i])

    # Create mutations from sampling
    mutated_sequences = []
    starting_pos = []
    wt_seq = []
    for sample in samples:
        sequence = list(wildtype)
        wt_seq.append(wildtype)
        starting_pos.append(positions[0])
        for i, mutation in enumerate(sample):
            sequence[positions[i]] = index_to_aminoacid[mutation]
        mutated_sequence = ''.join(sequence)
        mutated_sequences.append(mutated_sequence)
    input_df = pd.DataFrame({'wt_seq': wt_seq, 'mut_seq': mutated_sequences, 'start_pos': starting_pos})

    data = [
            ("protein1", wildtype),
        ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)

    scores = label_row_batch(input_df, wildtype, batch_tokens, model, alphabet)
    input_df['Score'] = scores

    return input_df['Score'].to_numpy()

def label_row(row, sequence, token_probs, alphabet):
    mut_list = np.array(list(row))
    wt_list = np.array(list(sequence))

    # Find positions where the lists are different
    idx = np.where(mut_list != wt_list)[0]
    mut_values = mut_list[idx]
    wt_values = wt_list[idx]

    scores = []
    for wt, mut, ind in zip(wt_values, mut_values, idx):
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mut)
        # add 1 for BOS
        score = token_probs[0, 1 + ind, mt_encoded] - token_probs[0, 1 + ind, wt_encoded]
        scores.append(score.item())
    scores = sum(scores)
    return scores

class Samples(Dataset):
    def __init__(self, x, sequence, batch_tokens, model, alphabet):
        self.x = x
        self.sequence = sequence
        self.batch_tokens = batch_tokens
        self.model = model
        self.alphabet = alphabet

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        row = self.x[idx]
        mut_list = np.array(list(row))
        wt_list = np.array(list(self.sequence))

        # Find positions where the lists are different
        idx_diff = np.where(mut_list != wt_list)[0]
        mut_values = mut_list[idx_diff]

        # Clone the batch_tokens tensor
        batch_tokens_masked_i = self.batch_tokens.clone()

        # Set mask_idx at appropriate positions
        batch_tokens_masked_i[0, 1 + idx_diff] = self.alphabet.mask_idx

        return batch_tokens_masked_i
    
def label_row_batch(df, sequence, batch_tokens, model, alphabet):

    x = df['mut_seq']
    batch_size = 512  
    batch_loader = Samples(x=df['mut_seq'], sequence=sequence, batch_tokens=batch_tokens, model=model, alphabet=alphabet)
    data_loader = DataLoader(batch_loader, batch_size=batch_size, shuffle=False)
    mut_token_probs = []
    with torch.no_grad():
        for batch_tokens_masked in tqdm(data_loader, desc="Processing batches"):
            batch_tokens_masked = batch_tokens_masked.view(batch_tokens_masked.shape[0], -1)
            mut_token_probs_i = torch.log_softmax(
                model(batch_tokens_masked.cuda())["logits"], dim=-1
            )
            mut_token_probs.append(mut_token_probs_i)
    mut_token_probs = torch.cat(mut_token_probs, dim=0)

    scores_all = []
    for i, row in enumerate(x):
        mut_list = np.array(list(row))
        wt_list = np.array(list(sequence))

        # Find positions where the lists are different
        idx = np.where(mut_list != wt_list)[0]
        mut_values = mut_list[idx]
        wt_values = wt_list[idx]

        scores = []
        for wt, mut, ind in zip(wt_values, mut_values, idx):
            wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mut)
            # add 1 for BOS
            score = mut_token_probs[i, 1 + ind, mt_encoded] - mut_token_probs[i, 1 + ind, wt_encoded]
            scores.append(score.item())
        scores_all.append(sum(scores))
    return scores_all

def label_row_singles(row, sequence, token_probs, batch_tokens, model, alphabet):
    mut_list = np.array(list(row))
    wt_list = np.array(list(sequence))

    # Find positions where the lists are different
    idx = np.where(mut_list != wt_list)[0]
    mut_values = mut_list[idx]
    wt_values = wt_list[idx]
    # print(mut_values)

    mut_token_probs = []
    # print(wt_values)
    # print(mut_values)
    # print(idx)
    for wt, mut, ind in zip(wt_values, mut_values, idx):
        # print('1')
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, 1 + ind] = alphabet.mask_idx
    # print('2')
    with torch.no_grad():
        mut_token_probs = torch.log_softmax(
            model(batch_tokens_masked.cuda())["logits"], dim=-1
        )
        # print("Shape of mut:", mut_token_probs.shape)

    scores = []
    for wt, mut, ind in zip(wt_values, mut_values, idx):
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mut)
        # add 1 for BOS
        score = mut_token_probs[0, 1 + ind, mt_encoded] - mut_token_probs[0, 1 + ind, wt_encoded]
        scores.append(score.item())
    scores = sum(scores)
    return scores