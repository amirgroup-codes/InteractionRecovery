SCRIPT_PATH="../generate_files.py"

# Generates files needed for q-sft. Input the fasta file path to the wild-type sequence, and the protein name
python "$SCRIPT_PATH" \
    --fasta_path "../fasta/GB1.fasta" \
    --protein_name "GB1"