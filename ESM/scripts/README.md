# Run scripts

## 0_setup.sh

Generates files needed for q-sft. Calls `generate_files.py`. Input the fasta file path to the wild-type sequence, and the protein name. Files for GB1, GFP, and TP53 have already been generated.

```console
python "$SCRIPT_PATH" \
    --fasta_path "../fasta/GB1.fasta" \
    --protein_name "GB1"
```

```console
sh 0_setup.sh
```

## 1_helper.sh

Run q-sft. Calls `run_helper.py`. A description of all parameters can be found below, as well as in `run_helper.py`. Results of experiments have been left in `results/GB1`.

```console
python "$HELPER_PATH" \
    --q 20 \
    --n 10 \
    --b 4 \
    --noise_sd 37.5 \
    --num_subsample 2 \
    --num_repeat 2 \
    --protein "GB1" \
    --region "randomcoils" \
    --random "False" \
    --hyperparam "False"
```

```console
sh 1_helper.sh
```

# q-sft parameters

## Setup
- **q:** Number of possible amino acids
- **n:** Number of sequences that are mutated
- **b:** Inner dimension of subsampling
- **noise_sd:** Hyperparameter: Proxy for the amount of additive noise in the signal
  - Optimal values: randomcoils = `37.5`, empiricalsamples = `43.3`
- **num_subsample, num_repeat:** control the amount of samples needed to run q-sft
- **protein:** protein to compute on (defaults are `GB1`, `GFP`, and `TP53`)
- **region:** optional parameter - designates folder in `results` where outputs get saved (defaults are `randomcoils` and `empiricalsites`)
- **random == True:** randomly select positions for sampling (0-indexed)
- **random == False:** manually input positions (1-indexed)
    - Random coils: positions = `[12, 15, 6, 47, 40, 9, 37, 50, 21, 38]`
    - Empirical sites: positions = `[41, 39, 31, 43, 27, 30, 54, 38, 23, 33]`
    - This can be specified at line 149 of `compute_samples.py`
- **hyperparam:** option to hyperparameter tune noise_sd