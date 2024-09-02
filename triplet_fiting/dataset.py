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
import numpy as np
import pandas as pd
from tqdm import tqdm


def form_datasets_triplets(df: pd.DataFrame, runs: list[int], p_ave_dbm: float,
                           M_shifted: int, M_orig: int, channel_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Forms datasets for triplet-based model fitting.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        runs (list[int]): List of run identifiers to include in the dataset.
        p_ave_dbm (float): Average power value in dBm.
        M_shifted (int): Number of neighboring points to consider for the shifted points.
        M_orig (int): Number of neighboring points to consider for the original points.
        channel_noise (bool): Whether to include channel noise in the dataset (default: False).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the dataset and the target dataset.
    """
    dataset = []
    target_dataset = []

    # Pre-fetch data for all runs
    data_for_runs = df[df.run.isin(runs) & (df.p_ave_dbm == p_ave_dbm)]
    scale_coef = df[(df.run == runs[0]) & (df.p_ave_dbm == p_ave_dbm)].scale_coef.values[0]

    for _, row in tqdm(data_for_runs.iterrows()):
        if channel_noise:
            points_shifted = (row['points_shifted'][0] - row['points_shifted_wo_noise'][0] + row['points_orig'][0]) * scale_coef  # here shifted poins are with noise and nonlinear effects, while wo_noise are without noise but with nonlinear effects. We need to add original points to get only channel noise effects for each constellation point (they will be at their original positions as constellation)
        else:
            points_shifted = row['points_shifted'][0] * scale_coef  # here shifted points has only nonlinear effects (no channel noise)
        points_orig = row['points_orig'][0] * scale_coef

        # Use Numpy for efficient extraction of neighbors
        extended_points_shift = np.concatenate([points_shifted[-M_shifted:], points_shifted, points_shifted[:M_shifted]])
        extended_points_orig = np.concatenate([points_orig[-M_orig:], points_orig, points_orig[:M_orig]])

        for i in range(len(points_shifted)):

            neighbors_shift = extended_points_shift[i:i + 2 * M_shifted + 1]

            # Interleave real and imaginary parts
            interleaved = np.empty(2 * len(neighbors_shift), dtype=neighbors_shift.dtype)
            interleaved[0::2] = neighbors_shift.real
            interleaved[1::2] = neighbors_shift.imag

            dataset.append(interleaved)

            neighbors_orig = extended_points_orig[i:i + 2 * M_orig + 1]

            # Interleave real and imaginary parts
            interleaved = np.empty(2 * len(neighbors_orig), dtype=neighbors_orig.dtype)
            interleaved[0::2] = neighbors_orig.real
            interleaved[1::2] = neighbors_orig.imag

            target_dataset.append(interleaved)

    return np.array(dataset), np.array(target_dataset)