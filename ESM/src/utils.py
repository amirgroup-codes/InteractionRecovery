import os
import re

def get_file_path(filename):
    path = os.path.join('src', filename)
    # custom_path = os.path.join(os.getcwd(), 'eve_exp', 'data')
    if os.path.exists(os.path.join(os.getcwd(), 'ESM', path)):
        file_path = os.path.join(os.getcwd(), path)
    elif os.path.exists(os.path.join(os.getcwd(), '..', path)):
        file_path = os.path.join(os.getcwd(), '..', path)
    elif os.path.exists(os.path.join(os.getcwd(), path)):
        file_path = os.path.join(os.getcwd(), path)
    elif os.path.exists(os.path.join(os.getcwd(), '..', '..', 'ESM', path)):
        file_path = os.path.join(os.getcwd(), '..', '..', 'ESM', path)
    else:
        file_path = os.path.join(os.getcwd(), '..', 'ESM', path)
    return file_path

def get_protein_path(filename, protein):
    path = os.path.join('src', protein, filename)
    if os.path.exists(os.path.join(os.getcwd(), 'ESM', path)):
        file_path = os.path.join(os.getcwd(), path)
    elif os.path.exists(os.path.join(os.getcwd(), '..', path)):
        file_path = os.path.join(os.getcwd(), '..', path)
    elif os.path.exists(os.path.join(os.getcwd(), path)):
        file_path = os.path.join(os.getcwd(), path)
    elif os.path.exists(os.path.join(os.getcwd(), '..', '..', 'ESM', path)):
        file_path = os.path.join(os.getcwd(), '..', '..', 'ESM', path)
    else:
        file_path = os.path.join(os.getcwd(), '..', 'ESM', path)
    return file_path

def extract_wildtype(file_path):
    with open(file_path, 'r') as file:
        fasta_content = file.read()
    match = re.search(r'\n([A-Za-z]+)', fasta_content)
    if match:
        sequence = match.group(1)
        return sequence
    else:
        return None
    
"""
ESM functions obtained from https://github.com/ntranoslab/esm-variants
"""
from Bio import SeqIO
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']

def fasta_to_dataframe(fasta_file):
    """
    Convert a FASTA file into a Pandas DataFrame.
    
    Parameters:
    fasta_file (str): Path of the FASTA file.
    
    Returns:
    df (DataFrame): Pandas DataFrame containing id, gene, seq, and length columns.
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    data = {
        "id": [record.id for record in records],
        "gene": [record.description.split()[1] if len(record.description.split()) > 1 else record.id for record in records],
        "seq": [str(record.seq) for record in records],
        "length": [len(record.seq) for record in records]
    }

    df = pd.DataFrame(data)

    return df

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

##### INFERENCE
def load_esm_model(model_name,device=1,parallel=False):
  repr_layer = int(model_name.split('_')[1][1:])
      
  # Check if the model is available locally
  directory = os.path.dirname(os.path.realpath(__file__))
  local_model_path = f"{directory}/../model/{model_name}.pth"
  
  if os.path.exists(local_model_path):
      # Load the model locally if it exists
      model = torch.load(local_model_path)
      alphabet = torch.load(f"{directory}/../model/{model_name}_alphabet.pth")
      batch_converter = alphabet.get_batch_converter()
  else:
      # Download the model using torch.hub
      model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
      batch_converter = alphabet.get_batch_converter()
      
      # Save the model locally
      torch.save(model, local_model_path)
      torch.save(alphabet, f"{directory}/../model/{model_name}_alphabet.pth")
    
  if parallel:
    device_ids = [1,4,5,6] if torch.cuda.is_available() else []
    model = torch.nn.DataParallel(model, device_ids=device_ids)

  return model.eval().to(device), alphabet, batch_converter, repr_layer
  # repr_layer = int(model_name.split('_')[1][1:])
  # model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
  # batch_converter = alphabet.get_batch_converter() 
  # return model.eval().to(device),alphabet,batch_converter,repr_layer

"""
FFT functions
"""
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

def sampling_function(samples, q, n, random_indices, wt_aa_dict, wildtype, positions, index_to_aminoacid, model, alphabet, batch_converter, protein_name, seq=None, AA_mutation=1):
    """Sampling Function."""

    # Convert 1-indexing to 0-indexing
    positions = [x - 1 for x in positions]
    random_indices = [x - 1 for x in random_indices]
    # Create random samples
    samples = samples.copy()
    # Mutate according to AA_mutation
    samples[samples == 1] = AA_mutation

    if protein_name == 'GFP_original_mutation' and q == 2 and n == 15:
        # Change to the specific mutations as specified in Poelwijk et. al
        aminoacid_to_index = {v: k for k, v in index_to_aminoacid.items()}
        for ind, AA in zip(positions, seq):
            wt_dict = wt_aa_dict[ind] 
            AA_number = aminoacid_to_index[AA]
            if AA_number == wt_dict[1]:
                continue
            else:
                value_of_1 = wt_dict[1]
                wt_dict[1] = AA_number
                for key, value in wt_dict.items():
                    if value == AA_number:
                        wt_dict[key] = value_of_1
        for i, ind in enumerate(random_indices):
            samples[:, i] = np.vectorize(wt_dict.get)(samples[:, i])
    else:
        for i, ind in enumerate(random_indices):
            wt_dict = wt_aa_dict[ind] 
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

"""
Edited functions to do batch sampling
"""
def get_PLL_batch(sequences, model, alphabet, batch_converter, reduce=np.sum, device=0):
    logits = get_logits_batch(sequences, model=model, batch_converter=batch_converter, device=device)
    first_seq_indices = [alphabet.tok_to_idx[i] for i in sequences[0]]
    indices = np.array([first_seq_indices] * len(sequences))
    logits_diagonal = logits[np.arange(logits.shape[0])[:, None], indices, indices]
    return reduce(logits_diagonal, axis=1)

def get_PLLR_batch(wt_seqs, mut_seqs, start_positions, model, alphabet, batch_converter, weighted=False, device=0):
    fn = np.sum if not weighted else np.mean
    wt_logits = get_PLL_batch(wt_seqs, model=model, alphabet=alphabet, batch_converter=batch_converter, reduce=fn, device=device)
    mut_logits = get_PLL_batch(mut_seqs, model=model, alphabet=alphabet, batch_converter=batch_converter, reduce=fn, device=device)
    return mut_logits - wt_logits

def get_logits_batch(sequences,model,batch_converter,format=None,device=0):
  data = [("_", seq) for seq in sequences]  
  _, _, batch_tokens = batch_converter(data)
  batch_tokens = batch_tokens.to(device)

  with torch.no_grad():
      logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"], dim=-1).cpu().numpy()
      token_probs = torch.log_softmax(model(batch_tokens.to(device))["logits"], dim=-1)
  if format=='pandas':
    WTlogits = pd.DataFrame(logits[0][1:-1,:],columns=alphabet.all_toks,index=list(seq)).T.iloc[4:24].loc[AAorder]
    WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
    return WTlogits
  else:
    return logits[:, 1:-1, :]
  
def get_wt_LLR(input_df,model,alphabet,batch_converter,device=0,silent=False): 
  # input: df.columns= id,	gene,	seq, length
  # make sure input_df does not contain any nonstandard amino acids
  AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']
  genes = input_df.id.values
  LLRs=[];input_df_ids=[]
  for gname in tqdm(genes,disable=silent):
    seq_length=input_df[input_df.id==gname].length.values[0]
    
    if seq_length<=1022:
      dt = [(gname+'_WT',input_df[input_df.id==gname].seq.values[0])]
      batch_labels, batch_strs, batch_tokens = batch_converter(dt)
      with torch.no_grad():
        results_ = torch.log_softmax(model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)['logits'],dim=-1)

      WTlogits = pd.DataFrame(results_[0,:,:].cpu().numpy()[1:-1,:],columns=alphabet.all_toks,index=list(input_df[input_df.id==gname].seq.values[0])).T.iloc[4:24].loc[AAorder]
      WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
      wt_norm=np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
      LLR = WTlogits - wt_norm

      LLRs += [LLR]
      input_df_ids+=[gname]

    else: ### tiling
      long_seq = input_df[input_df.id==gname].seq.values[0]
      ints,M,M_norm = get_intervals_and_weights(len(long_seq),min_overlap=512,max_len=1022,s=20)
      
      dt = []
      for i,idx in enumerate(ints):
        dt += [(gname+'_WT_'+str(i),''.join(list(np.array(list(long_seq))[idx])) )]

      logit_parts = []
      for dt_ in chunks(dt,20):
        batch_labels, batch_strs, batch_tokens = batch_converter(dt_)
        with torch.no_grad():
          results_ = torch.log_softmax(model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)['logits'],dim=-1)
        for i in range(results_.shape[0]):
          logit_parts += [results_[i,:,:].cpu().numpy()[1:-1,:]]
          
      logits_full = np.zeros((len(long_seq),33))
      for i in range(len(ints)):
        logit = np.zeros((len(long_seq),33))
        logit[ints[i]] = logit_parts[i].copy()
        logit = np.multiply(logit.T, M_norm[i,:]).T
        logits_full+=logit
        
      WTlogits = pd.DataFrame(logits_full,columns=alphabet.all_toks,index=list(input_df[input_df.id==gname].seq.values[0])).T.iloc[4:24].loc[AAorder]
      WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
      wt_norm=np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
      LLR = WTlogits - wt_norm

      LLRs += [LLR]
      input_df_ids+=[gname]

  return input_df_ids,LLRs

def get_logits(seq,model,batch_converter,format=None,device=0):
  data = [ ("_", seq),]
  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  batch_tokens = batch_tokens.to(device)
  with torch.no_grad():
      logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()
  if format=='pandas':
    WTlogits = pd.DataFrame(logits[0][1:-1,:],columns=alphabet.all_toks,index=list(seq)).T.iloc[4:24].loc[AAorder]
    WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
    return WTlogits
  else:
    return logits[0][1:-1,:]

def get_PLL(seq,model,alphabet,batch_converter,reduce=np.sum,device=0):
  s=get_logits(seq,model=model,batch_converter=batch_converter,device=device)
  idx=[alphabet.tok_to_idx[i] for i in seq]
  return reduce(np.diag(s[:,idx]))

### MELTed CSV
def meltLLR(LLR,savedir=None):
  vars = LLR.melt(ignore_index=False)
  vars['variant'] = [''.join(i.split(' '))+j for i,j in zip(vars['variable'],vars.index)]
  vars['score'] = vars['value']
  vars = vars.set_index('variant')
  vars['pos'] = [int(i[1:-1]) for i in vars.index]
  del vars['variable'],vars['value']
  if savedir is not None:
      vars.to_csv(savedir+'var_scores.csv')
  return vars

##################### TILING utils ###########################

def chop(L,min_overlap=511,max_len=1022):
  return L[max_len-min_overlap:-max_len+min_overlap]

def intervals(L,min_overlap=511,max_len=1022,parts=None):
  if parts is None: parts = []
  #print('L:',len(L))
  #print(len(parts))
  if len(L)<=max_len:
    if parts[-2][-1]-parts[-1][0]<min_overlap:
      #print('DIFF:',parts[-2][-1]-parts[-1][0])
      return parts+[np.arange(L[int(len(L)/2)]-int(max_len/2),L[int(len(L)/2)]+int(max_len/2)) ]
    else:
      return parts
  else:
    parts+=[L[:max_len],L[-max_len:]]
    L=chop(L,min_overlap,max_len)
    return intervals(L,min_overlap,max_len,parts=parts)

def get_intervals_and_weights(seq_len,min_overlap=511,max_len=1022,s=16):
  ints=intervals(np.arange(seq_len),min_overlap=min_overlap,max_len=max_len)
  ## sort intervals
  ints = [ints[i] for i in np.argsort([i[0] for i in ints])]

  a=int(np.round(min_overlap/2))
  t=np.arange(max_len)

  f=np.ones(max_len)
  f[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))
  f[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

  f0=np.ones(max_len)
  f0[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

  fn=np.ones(max_len)
  fn[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))

  filt=[f0]+[f for i in ints[1:-1]]+[fn]
  M = np.zeros((len(ints),seq_len))
  for k,i in enumerate(ints):
    M[k,i] = filt[k]
  M_norm = M/M.sum(0)
  return (ints, M, M_norm)


## PLLR score for indels
def get_PLLR(wt_seq,mut_seq,start_pos,model,alphabet,batch_converter,weighted=False,device=0):
  fn=np.sum if not weighted else np.mean
  if max(len(wt_seq),len(mut_seq))<=1022:
    return  get_PLL(mut_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn,device=device) - get_PLL(wt_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn,device=device)
  else:
    wt_seq,mut_seq,start_pos = crop_indel(wt_seq,mut_seq,start_pos)
    return  get_PLL(mut_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn,device=device) - get_PLL(wt_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn,device=device)

def crop_indel(ref_seq,alt_seq,ref_start):
  # Start pos: 1-indexed start position of variant
  left_pos = ref_start-1
  offset = len(ref_seq)-len(alt_seq)
  start_pos = int(left_pos - 1022 / 2)
  end_pos1 = int(left_pos + 1022 / 2) -min(start_pos,0)+ min(offset,0)
  end_pos2 = int(left_pos + 1022 / 2) -min(start_pos,0)- max(offset,0)
  if start_pos < 0: start_pos = 0 # Make sure the start position is not negative
  if end_pos1 > len(ref_seq): end_pos1 = len(ref_seq) # Make sure the end positions are not beyond the end of the sequence
  if end_pos2 > len(alt_seq): end_pos2 = len(alt_seq)
  if start_pos>0 and max(end_pos2,end_pos1) - start_pos <1022: ## extend to the left if there's space
            start_pos = max(0,max(end_pos2,end_pos1)-1022)
  
  return ref_seq[start_pos:end_pos1],alt_seq[start_pos:end_pos2],start_pos-ref_start

## stop gain variant score
def get_minLLR(seq,stop_pos):
  return min(get_wt_LLR(pd.DataFrame([('_','_',seq,len(seq))],columns=['id','gene','seq','length'] ),silent=True)[1][0].values[:,stop_pos:].reshape(-1))