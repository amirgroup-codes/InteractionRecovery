HELPER_PATH="../run_helper.py"

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
