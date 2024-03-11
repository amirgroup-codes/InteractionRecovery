import numpy as np
from group_lasso import GroupLasso
from sklearn.linear_model import Ridge
import time
from group_lasso._fista import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from qsft.utils import calc_hamming_weight, dec_to_qary_vec, qary_ints


@ignore_warnings(category=ConvergenceWarning)
def lasso_decode(signal, n_samples, noise_sd = 0, refine=True, verbose=False, report=True):
    """
    Implements Complex LASSO via Fast Iterative Soft Thresholding (FISTA) with optional Ridge Regression refinement
    Parameters
    ---------
    signal : Signal
    Signal object to be transformed.

    n_samples : int
    number of samples used in computing the transform.

    verbosity : bool
    If True printouts are increased.

    noise_sd : scalar
    Noise standard deviation.

    refine : bool
    If True Ridge Regression refinement is used.

    Returns
    -------
    gwht : dict
    Fourier transform (WHT) of the input signal

    runtime : scalar
    transform time + peeling time.

    locations : list
    List of nonzero indicies in the transform.

    avg_hamming_weight : scalar
    Average hamming wieght of non-zero indicies.

    max_hamming_weight : int
    Max hamming weight among the non-zero indicies.
    """
    q = signal.q
    n = signal.n
    N = q ** n
    dtype = int if (q ** 2)*n > 255 else np.uint8

    start_time = time.time()
    if verbose:
        print("Setting up LASSO problem")

    (sample_idx_dec, y) = list(signal.signal_t.keys()), list(signal.signal_t.values())

    sample_idx_dec = sample_idx_dec[:n_samples]
    y = y[:n_samples]

    #  WARNING: ADD NOISE ONLY FOR SYNTHETIC SIGNALS
    if signal.is_synt:
        y += np.random.normal(0, noise_sd / np.sqrt(2), size=(len(y), 2)).view(np.complex).reshape(len(y))

    sample_idx = dec_to_qary_vec(sample_idx_dec, q, n, dtype=dtype)
    y = np.concatenate((np.real(y), np.imag(y)))
    freqs = np.array(sample_idx).T @ qary_ints(n, q, dtype=dtype)
    X = np.exp(2j*np.pi*freqs/q).astype(np.csingle)
    X_ext = np.concatenate((np.concatenate((np.real(X), -np.imag(X)), axis=1), np.concatenate((np.imag(X), np.real(X)), axis=1)))
    groups = [i % N for i in range(2*N)]

    lasso_start = time.time()

    if verbose:
        print(f"Setup Time:{time.time() - start_time}sec")
        print("Running Iterations...")
        start_time = time.time()

    lasso = GroupLasso(groups=groups,
                       group_reg=0.1,
                       l1_reg=0,
                       tol=1e-8,
                       n_iter=25,
                       supress_warning=True,
                       fit_intercept=False)
    lasso.fit(X_ext, y)

    if verbose:
        print(f"LASSO fit time:{time.time() - start_time}sec")

    w = lasso.coef_

    non_zero = np.nonzero(w[:, 0])[0]

    if len(non_zero) > 0 and refine:
        ridge = Ridge(alpha=1e-2, tol=1e-8)
        ridge.fit(X_ext[:, non_zero], y)
        w[non_zero] = ridge.coef_[:, np.newaxis]
    gwht = w[0:N] + 1j*w[N:(2*N)]

    gwht = np.reshape(gwht, [q] * n)

    gwht_dict = {}

    non_zero_pos = np.array(np.nonzero(gwht)).T
    for p in non_zero_pos:
        gwht_dict[tuple(p)] = gwht[tuple(p)]

    runtime = time.time() - lasso_start

    if not report:
        return gwht_dict
    else:
        if len(non_zero_pos) > 0:
            loc = list(non_zero_pos)
            avg_hamming_weight = np.mean(calc_hamming_weight(loc))
            max_hamming_weight = np.max(calc_hamming_weight(loc))
        else:
            loc, avg_hamming_weight, max_hamming_weight = [], 0, 0

        result = {
            "gwht": gwht_dict,
            "n_samples": n_samples,
            "locations": loc,
            "runtime": runtime,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight
        }
        return result
