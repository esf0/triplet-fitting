"""
Copyright (c) 2024 Egor Sedov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

1. Attribution: You must give appropriate credit, provide a link to the license,
   and indicate if changes were made. You may do so in any reasonable manner, but
   not in any way that suggests the licensor endorses you or your use.

2. No Cloning without Citation: You are not allowed to clone this repository or
   any substantial part of it without citing the author. Citation must include
   the author's name (Egor Sedov) and contact email (egor.sedoff+git@gmail.com).

3. The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any, Optional
from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import matlab.engine
import hpcom.modulation
from tqdm import tqdm

from .utils import form_column_names
from .dataset import form_datasets_triplets
from .matlab_interface import start_matlab_engine


def fit_gauss_matlab(X: np.ndarray, H: int, plotlik: int = 0, plotsolution: int = 0, maxit: int = 50,
                     minDeterminant: float = 0.0001, eng: Optional[matlab.engine.MatlabEngine] = None) \
        -> dict[str, Any]:
    """
    Fits a Gaussian Mixture Model (GMM) using MATLAB's BRML toolkit.

    Args:
        X (np.ndarray): Input data to fit.
        H (int): Number of mixture components.
        plotlik (int): Whether to plot likelihood during fitting (default: 0).
        plotsolution (int): Whether to plot the solution (default: 0).
        maxit (int): Maximum number of iterations (default: 50).
        minDeterminant (float): Minimum determinant for covariance matrices (default: 0.0001).
        eng (Optional[matlab.engine.MatlabEngine]): MATLAB engine instance (default: None).

    Returns:
        dict[str, Any]: A dictionary containing the learned parameters:
            - 'P': Learned mixture coefficients.
            - 'm': Learned means.
            - 'S': Learned covariances.
            - 'eig': Eigenvalues of the covariance matrices.
            - 'loglik': Log-likelihood of the fitted model.
            - 'phgn': Mixture assignment probabilities.
    """

    brml_location = str(Path(__file__).parent.parent / "BRMLtoolkit") + "/"

    # Start MATLAB engine if not provided
    flag_start_end = False
    if eng is None:
        eng = start_matlab_engine()
        flag_start_end = True

    eng.addpath(brml_location, nargout=0)
    eng.addpath(brml_location + '+brml/', nargout=0)

    # Convert data to MATLAB format
    X_matlab = matlab.double(X.tolist())
    H_matlab = eng.double(H)

    # Set options for GMMem function
    opts = eng.struct()
    opts['plotlik'] = plotlik
    opts['plotsolution'] = plotsolution
    opts['maxit'] = maxit
    opts['minDeterminant'] = minDeterminant

    # Call GMMem function
    P, m, S, loglik, phgn = eng.brml.GMMem(X_matlab, H_matlab, opts, nargout=5)

    # Convert MATLAB outputs to numpy arrays
    P = np.asarray(P)
    m = np.asarray(m).T  # Transpose for consistency
    S = np.asarray(S)

    if H == 1:
        P = np.array([P])
        S = S.reshape((1, -1))
    else:
        P = P.reshape(-1)
        S = S.transpose(2, 0, 1).reshape(H, -1)

    phgn = np.asarray(phgn)

    # Terminate MATLAB engine if started locally
    if flag_start_end:
        eng.quit()

    # Calculate eigenvalues of the covariance matrices
    eigenvalues = np.array([np.linalg.eigvals(S_cur.reshape(2, 2)) for S_cur in S])

    return {
        'P': P,
        'm': m,
        'S': S,
        'eig': eigenvalues,
        'loglik': loglik,
        'phgn': phgn
    }


def fit_triplets(df: pd.DataFrame, path_to_save: str, matlab_engine: matlab.engine.MatlabEngine,
                 disp: bool = False, H_max: int = 1, independant_gauss: bool = False) -> pd.DataFrame:
    # Initialize an empty DataFrame
    df_result = pd.DataFrame()

    # Get the constellation for 16QAM
    constellation = hpcom.modulation.get_constellation('16qam')
    triplets = list(itertools.product(constellation, constellation, constellation))

    for triplet in tqdm(triplets):
        triplet_left_point = triplet[0]
        triplet_central_point = triplet[1]
        triplet_right_point = triplet[2]

        epsilon = 1e-10
        condition = (
            np.isclose(df.orig_m1_real, np.real(triplet_left_point), atol=epsilon) &
            np.isclose(df.orig_m1_imag, np.imag(triplet_left_point), atol=epsilon) &
            np.isclose(df.orig_real, np.real(triplet_central_point), atol=epsilon) &
            np.isclose(df.orig_imag, np.imag(triplet_central_point), atol=epsilon) &
            np.isclose(df.orig_p1_real, np.real(triplet_right_point), atol=epsilon) &
            np.isclose(df.orig_p1_imag, np.imag(triplet_right_point), atol=epsilon)
        )

        df_one = df[condition]

        row_data: dict[str, Any] = {
            'left_point_real': triplet_left_point.real,
            'left_point_imag': triplet_left_point.imag,
            'central_point_real': triplet_central_point.real,
            'central_point_imag': triplet_central_point.imag,
            'right_point_real': triplet_right_point.real,
            'right_point_imag': triplet_right_point.imag,
        }

        x = df_one.shifted_real.values - triplet_central_point.real
        y = df_one.shifted_imag.values - triplet_central_point.imag
        data_to_fit = np.stack((x, y), axis=0)

        # Loop over different values of H
        for H in range(1, H_max + 1):
            result = fit_gauss_matlab(data_to_fit, H=H, plotlik=0, plotsolution=0, maxit=50,
                                      minDeterminant=0.0001, eng=matlab_engine)

            # Extract parameters
            P = result['P']
            m = result['m']
            S = result['S']
            eig = result['eig']
            loglik = result['loglik']

            # Add the log-likelihood to the row
            row_data[f'h{H}_loglik'] = loglik

            # Add the parameters for each Gaussian component
            for i in range(H):
                row_data[f'h{H}_prob_{i+1}'] = P[i]
                row_data[f'h{H}_mean0_{i+1}'] = m[i, 0]
                row_data[f'h{H}_mean1_{i+1}'] = m[i, 1]
                row_data[f'h{H}_cov00_{i+1}'] = S[i, 0]
                row_data[f'h{H}_cov01_{i+1}'] = S[i, 1]
                row_data[f'h{H}_cov10_{i+1}'] = S[i, 2]
                row_data[f'h{H}_cov11_{i+1}'] = S[i, 3]
                row_data[f'h{H}_eig0_{i+1}'] = eig[i, 0]
                row_data[f'h{H}_eig1_{i+1}'] = eig[i, 1]

        if independant_gauss:
            energy_noise_for_points = np.mean(np.abs(x + 1j * y) ** 2) / 2
            sigma_for_points = np.sqrt(energy_noise_for_points)
            row_data['sigma_independent_gauss'] = sigma_for_points

        # Append the row to the DataFrame
        df_result = pd.concat([df_result, pd.DataFrame([row_data])], ignore_index=True)

    # Save results
    df_result.to_pickle(path_to_save)
    df_result.to_parquet(path_to_save[:-3] + 'parquet', engine='pyarrow')

    return df_result


def fit_all(p_ave_dbm_list: list[float], data_dir: str, filename_prefix: str, n_runs: int = 64,
            H_max: int = 1, independant_gauss: bool = False, channel_noise: bool = False) -> pd.DataFrame:
    eng = matlab.engine.start_matlab()

    for p_ave_dbm in p_ave_dbm_list:
        filename = f"{filename_prefix}{p_ave_dbm}.pkl"
        df = pd.read_pickle(f"{data_dir}/{filename}")

        print('Forming datasets')
        runs = [k for k in range(n_runs)]  # Replace with your list of runs
        M = 1  # Number of neighbors on each side
        dataset, target_dataset = form_datasets_triplets(df, runs, p_ave_dbm, M, M, channel_noise=channel_noise)
        df_new = pd.concat([pd.DataFrame(dataset), pd.DataFrame(target_dataset)], axis=1).astype(float)
        df_new.columns = form_column_names(M, M)

        print('Fitting triplets')
        df_result = fit_triplets(df_new, path_to_save=f"{data_dir}/{filename_prefix}{p_ave_dbm}_fit.pkl",
                                 H_max=H_max, matlab_engine=eng, disp=False, independant_gauss=independant_gauss)

        print(f'Finished {filename_prefix} for p_ave_dbm = {p_ave_dbm} dBm')

    eng.quit()
    return df_result