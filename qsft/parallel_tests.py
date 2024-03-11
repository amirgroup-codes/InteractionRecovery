import itertools
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd

from qsft.test_helper import TestHelper
tqdm = partial(tqdm, position=0, leave=True)


def _test(i):
    """
    Runs a single instance of a test
    Parameters
    ----------
    i
    index of the test
    Returns
    -------
    result : dict
    Result of a test, containing a variety of important quantities
    """
    df_row = test_df.iloc[i]
    num_subsample, num_repeat, b, noise_sd = int(df_row["num_subsample"]), int(df_row["num_repeat"]), int(df_row["b"]), df_row["noise_sd"]

    # set model arguments
    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    model_kwargs["noise_sd"] = noise_sd
    model_kwargs["n_samples"] = num_subsample * (helper_obj.q ** b) * num_repeat * (helper_obj.n + 1)
    model_result = helper_obj.compute_model(method=method, model_kwargs=model_kwargs, report=True, verbosity=0)

    test_kwargs = {}
    test_kwargs["beta"] = model_result.get("gwht")

    del model_result['gwht']
    del model_result['locations']

    nmse = helper_obj.test_model(method=method, **test_kwargs)

    result = {}
    result["n"] = helper_obj.n
    result["q"] = helper_obj.q
    result["runtime"] = model_result.get("runtime")
    result["found_sparsity"] = len(test_kwargs["beta"])
    result["n_samples"] = model_result.get("n_samples")
    result["ratio_samples"] = model_result.get("n_samples") / (helper_obj.q ** helper_obj.n)
    result["max_hamming_weight"] = model_result.get("max_hamming_weight")
    result["nmse"] = nmse
    result["method"] = method

    return result


def run_tests(test_method, helper: TestHelper, iters, num_subsample_list, num_repeat_list, b_list, noise_sd_list, parallel=True):
    """
    Run an experiment based on the given parameters
    Parameters
    ----------
    test_method
    helper
    iters
    num_subsample_list
    num_repeat_list
    b_list
    noise_sd_list
    parallel

    Returns
    -------

    """
    global test_df
    global helper_obj
    global method

    helper_obj = helper
    method = test_method

    test_params_list = list(itertools.product(num_subsample_list, num_repeat_list, b_list, noise_sd_list, range(iters)))
    test_df = pd.DataFrame(data=test_params_list, columns=["num_subsample", "num_repeat", "b", "noise_sd", "iter"])

    exp_count = len(test_df)

    pred = []

    if parallel:
        with Pool() as pool:
            # run the tests in parallel
            pbar = tqdm(pool.imap(_test, range(exp_count)), total=exp_count)
            best_result = np.inf
            for result in pbar:
                pred.append(result)
                best_result = min(best_result, result['nmse'])
                pbar.set_postfix({"min NMSE": best_result})
    else:
        pbar = tqdm(range(exp_count))
        best_result = np.inf
        for i in pbar:
            result = _test(i)
            pred.append(result)
            if result['nmse'] < best_result:
                best_result = result['nmse']
                df_row = test_df.iloc[i]
                num_subsample, num_repeat, b = int(df_row["num_subsample"]), int(df_row["num_repeat"]), int(df_row["b"])
                pbar.set_postfix({"min NMSE": best_result, "b": b, "C": num_subsample, "P1": num_repeat})

    results_df = pd.DataFrame(data=pred)
    results_df = pd.concat([test_df, results_df], axis=1, join="inner")

    return results_df
