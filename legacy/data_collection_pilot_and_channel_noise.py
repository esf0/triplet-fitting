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
# Import of necessary packages
import tensorflow as tf  # tensorflow used for GPU memory allocation
import pandas as pd  # pandas used for data storage
import hpcom  # hpcom used for signal generation and channel modelling
from tqdm import tqdm  # tqdm used for progress bar

from hpcom.signal import (create_wdm_parameters, generate_wdm, generate_wdm_optimise, receiver, receiver_wdm,
                     nonlinear_shift, dbm_to_mw, get_default_wdm_parameters, get_points_wdm,
                     generate_ofdm_signal, decode_ofdm_signal,
                     generate_wdm_new)
from hpcom.modulation import get_modulation_type_from_order, get_scale_coef_constellation, \
    get_nearest_constellation_points_unscaled, get_constellation, get_nearest_constellation_points_new
from hpcom.metrics import get_ber_by_points, get_ber_by_points_ultimate, get_energy, get_average_power, get_evm_ultimate, \
    get_evm, calculate_mutual_information

from ssfm_gpu.propagation import propagate_manakov, propagate_manakov_backward, \
    propagate_schrodinger, dispersion_compensation_manakov, dispersion_compensation

from datetime import datetime
import numpy as np
import scipy as sp
from math import ceil

# Directory with data files for Linux and Windows
# data_dir = "/home/username/data/"
# data_dir = 'C:/Users/username/data/'
data_dir = 'C:/Users/190243539/PycharmProjects/nn_essential/benchmark/data/errorstat_channel_noise/'


# Name of the job to store data for different parameters
job_name = 'errorstat_onepol_pilot_channel_noise_2'

# System parameters
GPU_MEM_LIMIT = 1024 * 7  # 6 GB of GPU memory is allocated

# Signal parameters
n_polarisations = 1  # number of polarisations. Can be 1 for NLSE and 2 for Manakov
n_symbols = 2 ** 18  # number of symbols to be transmitted for each run
m_order = 16  # modulation order (16-QAM, 64-QAM, etc.)
symb_freq = 34e9  # symbol frequency
channel_spacing = 75e9  # channel spacing
roll_off = 0.1  # roll-off factor for RRC filter
upsampling = 16  # upsampling factor
downsampling_rate = 1  # downsampling rate
# p_ave_dbm_list = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # list of average power values in dBm
# p_ave_dbm_list = [5, 6, 7, 8]  # list of average power values in dBm
# p_ave_dbm_list = [-1.5, -1.4, -1.3, -1.25, -1.2, -1.1, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5]  # list of average power values in dBm
p_ave_dbm_list = [-1.5, -1.25, 1.25, 1.5]  # list of average power values in dBm

# Channel parameters
z_span = 80  # span length in km
# n_channels_list = [1]  # number of WDM channels
# n_span_list = [15]  # 480, 640, 800, 960, 1120
# noise_figure_db_list = [4.5]  # list of noise figure values in dB. -200 means no noise
alpha_db = 0.2  # attenuation coefficient in dB/km
gamma = 1.2  # nonlinearity coefficient
dispersion_parameter = 16.8  # dispersion parameter in ps/nm/km
dz = 1  # step size in km

# Simulation parameters
n_runs = 2 ** 7  # number of runs for each parameter set
verbose = 0  # verbose level. 0 - no print, 3 - print all system logs
# seed = 'fixed'  # seed for random number generator. 'time' - use current time, 'fixed' - fixed seed
channels_type = 'middle'  # type for which of WDM channels all metrics will be calculated.
# 'middle' - middle channel, 'all' - all channels

pilot = (1 + 1j)
pilot_n = 0
pilot_freq_n = 100


n_channels = 1
# p_ave_dbm = 8
noise_figure_db = 4.5
n_span = 15


# GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus), gpus)
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



def get_noise_sigma_points_shifted(points_shifted, points_shifted_wo_noise):
    energy_noise_for_points = np.mean(np.abs(points_shifted - points_shifted_wo_noise) ** 2) / 2
    sigma_for_points = np.sqrt(energy_noise_for_points)
    noise_for_points = np.random.normal(0, sigma_for_points, len(points_shifted)) + \
                       1j * np.random.normal(0, sigma_for_points, len(points_shifted))
    points_shifted_w_noise = points_shifted_wo_noise + noise_for_points

    return sigma_for_points, points_shifted_w_noise


def get_noise_sigma_points(points, points_wo_noise, points_orig):
    nl_shift = nonlinear_shift(points, points_orig)
    points_shifted = points * nl_shift

    nl_shift_wo_noise = nonlinear_shift(points_wo_noise, points_orig)
    points_shifted_wo_noise = points_wo_noise * nl_shift_wo_noise

    return get_noise_sigma_points_shifted(points_shifted, points_shifted_wo_noise)


def magic(wdm, channel, channel_wo_noise):


    # generate points with pilot symbols
    # pilots = pilot * np.ones(ceil(wdm['n_symbols'] / pilot_freq_n), dtype=complex)
    seed = datetime.now().timestamp()
    bits = hpcom.signal.gen_wdm_bit_sequence(wdm['n_symbols'], wdm['modulation_type'],
                                             n_carriers=1, seed=seed)

    points_gen = hpcom.modulation.get_constellation_point(bits, type=wdm['modulation_type'])
    mod_type = hpcom.modulation.get_modulation_type_from_order(wdm['m_order'])
    scale_constellation = np.sqrt(wdm['p_ave']) / hpcom.modulation.get_scale_coef_constellation(mod_type)
    for k in range(pilot_n):
        points_gen[k::pilot_freq_n] = pilot
    points_gen = points_gen * scale_constellation

    # tf.random.set_seed(0)  # set seed for channel propagation (noise)
    tf.random.set_seed(seed)


    # calculate timestep
    dt = 1. / wdm['sample_freq']

    # generate signal (one or two polarisations)
    # signal contains one or two elements which is x and y polarisations
    signal, wdm_info = generate_wdm_new(wdm, bits=None, points=([points_gen], [points_gen]), ft_filter_values=None)
    # signal, wdm_info = generate_wdm_new(wdm, bits=None, points=None, ft_filter_values=None)

    # same as signal. points_orig[0] contains points for x polarisation, points_orig[1] for y polarisation (if exist)
    points_orig = wdm_info['points']

    ft_filter_values = wdm_info['ft_filter_values']

    np_signal = len(signal[0])

    e_signal_x = get_energy(signal[0], dt * np_signal)
    p_signal_x = get_average_power(signal[0], dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / wdm['n_polarisations'] * wdm['n_channels']

    if wdm['n_polarisations'] == 2:
        e_signal_y = get_energy(signal[1], dt * np_signal)
        p_signal_y = get_average_power(signal[1], dt)
        mes = (f"Average signal power (x / y): {p_signal_x:.7f} "
               f"/ {p_signal_y:.7f} (has to be close to {p_signal_correct:.7f})\n"
               f"Average signal energy (x / y): {e_signal_x:.7f} / {e_signal_y:.7f}")
    else:
        mes = (f"Average signal power (x): {p_signal_x:.7f} "
               f"(has to be close to {p_signal_correct:.7f})\n"
               f"Average signal energy (x): {e_signal_x:.7f}")

    print(mes) if verbose >= 3 else ...

    start_time = datetime.now()
    if wdm['n_polarisations'] == 1:
        signal_prop = propagate_schrodinger(channel, signal[0], wdm['sample_freq'])
        signal_cdc = dispersion_compensation(channel, signal_prop, dt)
        signal_prop = (signal_prop,)  # create tuple to use [0] index for only one polarisation
        signal_cdc = (signal_cdc,)

        signal_prop_wo_noise = propagate_schrodinger(channel_wo_noise, signal[0], wdm['sample_freq'])
        signal_cdc_wo_noise = dispersion_compensation(channel_wo_noise, signal_prop_wo_noise, dt)
        signal_prop_wo_noise = (signal_prop_wo_noise,)  # create tuple to use [0] index for only one polarisation
        signal_cdc_wo_noise = (signal_cdc_wo_noise,)
    else:
        signal_prop = propagate_manakov(channel, signal[0], signal[1], wdm['sample_freq'])
        signal_cdc = dispersion_compensation_manakov(channel, signal_prop[0], signal_prop[1], dt)

        signal_prop_wo_noise = propagate_manakov(channel_wo_noise, signal[0], signal[1], wdm['sample_freq'])
        signal_cdc_wo_noise = dispersion_compensation_manakov(channel_wo_noise, signal_prop_wo_noise[0],
                                                              signal_prop_wo_noise[1], dt)

    print("propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    e_signal_x_prop = get_energy(signal_prop[0], dt * np_signal)
    e_signal_x_prop_wo_noise = get_energy(signal_prop_wo_noise[0], dt * np_signal)
    if wdm['n_polarisations'] == 2:
        e_signal_y_prop = get_energy(signal_prop[1], dt * np_signal)
        mes = (f"Average signal energy after propagation (x / y): {e_signal_x_prop:.7f} / {e_signal_y_prop:.7f} \n"
               f"Energy difference (x / y): {np.absolute(e_signal_x_prop - e_signal_x):.7f} / "
               f"{np.absolute(e_signal_y_prop - e_signal_y):.7f}")
    else:
        mes = (f"Average signal energy after propagation (x): {e_signal_x_prop:.7f} \n"
               f"Energy difference (x): {np.absolute(e_signal_x_prop - e_signal_x):.7f}")

    print(mes) if verbose >= 3 else ...

    # check noise
    noise_trace = signal_prop[0] - signal_prop_wo_noise[0]
    e_noise = get_energy(noise_trace, dt * np_signal)
    print(f"Average noise energy (x): {e_noise:.7f}") if verbose >= 3 else ...

    snr = e_signal_x_prop_wo_noise / e_noise
    sigma = np.sqrt(e_noise) / 2

    # Generate Gaussian noise
    noise = np.random.normal(0, sigma, np_signal) + 1j * np.random.normal(0, sigma, np_signal)

    # Calculate energy of the noise
    # noise_energy = np.sum(np.abs(noise)**2)
    noise_energy = get_energy(noise, dt * np_signal)
    print(f"Average noise energy (simulated) (x): {noise_energy:.7f}") if verbose >= 3 else ...

    mod_type = get_modulation_type_from_order(wdm['m_order'])
    constellation = get_constellation(mod_type)

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    samples = []
    samples_wo_noise = []

    points = []
    points_shifted = []
    points_found = []
    ber = []
    q = []
    evm = []
    mi = []

    points_wo_noise = []
    points_shifted_wo_noise = []
    points_found_w_noise = []
    ber_w_noise = []
    q_w_noise = []
    evm_w_noise = []
    mi_w_noise = []

    for p in range(wdm['n_polarisations']):

        print('Polarisation ', p) if verbose >= 1 else ...
        samples.append(receiver_wdm(signal_cdc[p], ft_filter_values[p], wdm))
        samples_wo_noise.append(receiver_wdm(signal_cdc_wo_noise[p], ft_filter_values[p], wdm))

        # TODO: make CDC after receiver
        # TODO: handle dbp
        # for k in range(wdm['n_channels']):
        #     samples_x[k], samples_y[k] = dispersion_compensation(channel, samples_x[k], samples_y[k], wdm['downsampling_rate'] / wdm['sample_freq'])

        points_per_pol = []
        points_shifted_per_pol = []
        points_found_per_pol = []
        ber_per_pol = []
        q_per_pol = []
        evm_per_pol = []
        mi_per_pol = []

        points_per_pol_wo_noise = []
        points_shifted_per_pol_wo_noise = []
        points_shifted_per_pol_w_noise = []
        points_found_per_pol_w_noise = []
        ber_per_pol_w_noise = []
        q_per_pol_w_noise = []
        evm_per_pol_w_noise = []
        mi_per_pol_w_noise = []

        for k in range(wdm['n_channels']):
            if channels_type == 'middle' and k != (wdm['n_channels'] - 1) // 2:
                continue

            print('WDM channel', k) if verbose >= 1 else ...

            points_per_pol.append(get_points_wdm(samples[p][k], wdm))
            points_per_pol_wo_noise.append(get_points_wdm(samples_wo_noise[p][k], wdm))

            nl_shift = nonlinear_shift(points_per_pol[-1], points_orig[p][k])
            points_shifted_per_pol.append(points_per_pol[-1] * nl_shift)
            nl_shift_wo_noise = nonlinear_shift(points_per_pol_wo_noise[-1], points_orig[p][k])
            points_shifted_per_pol_wo_noise.append(points_per_pol_wo_noise[-1] * nl_shift_wo_noise)

            # Calculate the energy of the noise-free signal
            noise_points_orig = points_shifted_per_pol[-1] - points_shifted_per_pol_wo_noise[-1]
            # energy_noise_points = np.sum(np.abs(points_shifted_per_pol[-1] - points_shifted_per_pol_wo_noise[-1])**2)
            energy_noise_points = np.mean(np.abs(noise_points_orig) ** 2) / 2
            sigma_points = np.sqrt(energy_noise_points)
            noise_points = np.random.normal(0, sigma_points, len(points_shifted_per_pol[-1])) + \
                           1j * np.random.normal(0, sigma_points, len(points_shifted_per_pol[-1]))
            points_shifted_per_pol_w_noise.append(points_shifted_per_pol_wo_noise[-1] + noise_points)
            print(f"Sigma (points): {sigma_points:.7f}") if verbose >= 3 else ...
            print(f"Scale coef: {wdm['scale_coef']}") if verbose >= 3 else ...

            points_orig_scaled = points_orig[p][k] * wdm['scale_coef']

            start_time = datetime.now()
            points_found_per_pol.append(
                get_nearest_constellation_points_new(points_shifted_per_pol[-1] * wdm['scale_coef'], constellation)
                # get_nearest_constellation_points_unscaled(points_shifted_per_pol[-1], mod_type)
            )
            points_found_per_pol_w_noise.append(
                get_nearest_constellation_points_new(points_shifted_per_pol_w_noise[-1] * wdm['scale_coef'],
                                                     constellation)
            )
            print(
                f"search {p} polarisation {k} channel points took {(datetime.now() - start_time).total_seconds() * 1000} ms"
            ) if verbose >= 2 else ...

            start_time = datetime.now()
            ber_per_pol.append(get_ber_by_points(points_orig_scaled, points_found_per_pol[-1], mod_type))
            q_per_pol.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_per_pol[-1][0]))
            evm_per_pol.append(get_evm(points_orig_scaled, points_shifted_per_pol[-1] * wdm['scale_coef']))
            mi_per_pol.append(calculate_mutual_information(points_orig_scaled, points_found_per_pol[-1]))
            print(
                f"search {p} polarisation {k} channel metrics took {(datetime.now() - start_time).total_seconds() * 1000} ms"
            ) if verbose >= 2 else ...

            start_time = datetime.now()
            ber_per_pol_w_noise.append(
                get_ber_by_points(points_orig_scaled, points_found_per_pol_w_noise[-1], mod_type))
            q_per_pol_w_noise.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_per_pol_w_noise[-1][0]))
            evm_per_pol_w_noise.append(
                get_evm(points_orig_scaled, points_shifted_per_pol_w_noise[-1] * wdm['scale_coef']))
            mi_per_pol_w_noise.append(
                calculate_mutual_information(points_orig_scaled, points_found_per_pol_w_noise[-1]))
            print(
                f"search {p} polarisation {k} channel metrics took {(datetime.now() - start_time).total_seconds() * 1000} ms"
            ) if verbose >= 2 else ...

        points.append(points_per_pol)
        points_shifted.append(points_shifted_per_pol)
        points_found.append(points_found_per_pol)
        ber.append(ber_per_pol)
        q.append(q_per_pol)
        evm.append(evm_per_pol)
        mi.append(mi_per_pol)

        points_wo_noise.append(points_per_pol_wo_noise)
        points_shifted_wo_noise.append(points_shifted_per_pol_wo_noise)
        points_found_w_noise.append(points_found_per_pol_w_noise)
        ber_w_noise.append(ber_per_pol_w_noise)
        q_w_noise.append(q_per_pol_w_noise)
        evm_w_noise.append(evm_per_pol_w_noise)
        mi_w_noise.append(mi_per_pol_w_noise)

    if channels_type == 'middle':
        k_range = [0]
    else:
        k_range = range(wdm['n_channels'])

    for k in k_range:

        ber_text = f"BER (x / y): {ber[0][k]} / {ber[1][k]}" if wdm['n_polarisations'] == 2 else f"BER (x): {ber[0][k]}"
        q_text = f"Q^2-factor (x / y): {q[0][k]} / {q[1][k]}" if wdm[
                                                                     'n_polarisations'] == 2 else f"Q^2-factor (x): {q[0][k]}"
        evm_text = f"EVM (x / y): {evm[0][k]} / {evm[1][k]}" if wdm['n_polarisations'] == 2 else f"EVM (x): {evm[0][k]}"
        mi_text = f"MI (x / y): {mi[0][k]} / {mi[1][k]}" if wdm['n_polarisations'] == 2 else f"MI (x): {mi[0][k]}"

        if verbose >= 1:
            print('WDM channel (if channels_type == "middle", 0 for middle channel)', k)
            print(ber_text)
            print(q_text)
            print(evm_text)
            print(mi_text)

        result = {
            'points': points,
            'points_orig': points_orig,
            'points_shifted': points_shifted,
            'points_found': points_found,
            'ber': ber,
            'q': q,
            'evm': evm,
            'mi': mi,
            'points_wo_noise': points_wo_noise,
            'points_shifted_wo_noise': points_shifted_wo_noise,
            'points_found_w_noise': points_found_w_noise,
            'ber_w_noise': ber_w_noise,
            'q_w_noise': q_w_noise,
            'evm_w_noise': evm_w_noise,
            'mi_w_noise': mi_w_noise,
        }

    return result


# print(f'n_channels = {n_channels} / '
#               f'p_dbm = {p_ave_dbm} / n_span = {n_span} / '
#               f'noise = {noise_figure_db}')

start_time = datetime.now()

for p_ave_dbm in p_ave_dbm_list:
    # Create an empty dataframe
    df = pd.DataFrame()

    start_time_pave = datetime.now()
    print(f'Power: {p_ave_dbm} dBm / Noise: {noise_figure_db} dB / Span: {n_span} spans')
    for run in tqdm(range(n_runs)):


        # Signal parameters
        wdm = hpcom.signal.create_wdm_parameters(n_channels=n_channels, p_ave_dbm=p_ave_dbm,
                                                 n_symbols=n_symbols, m_order=m_order,
                                                 roll_off=roll_off, upsampling=upsampling,
                                                 downsampling_rate=downsampling_rate,
                                                 symb_freq=symb_freq,
                                                 channel_spacing=channel_spacing,
                                                 n_polarisations=n_polarisations, seed='fixed')

        # Channel parameters
        channel = hpcom.channel.create_channel_parameters(n_spans=n_span,
                                                          z_span=z_span,
                                                          alpha_db=alpha_db,
                                                          gamma=gamma,
                                                          noise_figure_db=noise_figure_db,
                                                          dispersion_parameter=dispersion_parameter,
                                                          dz=dz)

        channel_wo_noise = hpcom.channel.create_channel_parameters(n_spans=n_span,
                                                                   z_span=z_span,
                                                                   alpha_db=alpha_db,
                                                                   gamma=gamma,
                                                                   noise_figure_db=-200,
                                                                   dispersion_parameter=dispersion_parameter,
                                                                   dz=dz)

        result = magic(wdm, channel, channel_wo_noise)

        result_dict = {}
        result_dict['run'] = run
        result_dict['n_channels'] = n_channels
        result_dict['n_polarisations'] = n_polarisations
        result_dict['n_symbols'] = n_symbols
        result_dict['p_ave_dbm'] = p_ave_dbm
        result_dict['z_km'] = n_span * 80
        result_dict['scale_coef'] = wdm['scale_coef']

        result_dict['noise_figure_db'] = channel['noise_figure_db']
        result_dict['gamma'] = channel['gamma']
        result_dict['z_span'] = channel['z_span']
        result_dict['dispersion_parameter'] = channel['dispersion_parameter']
        result_dict['dz'] = channel['dz']

        result_dict['points_orig'] = result['points_orig']
        # result_dict['points'] = result['points']
        result_dict['points_shifted'] = result['points_shifted']

        # result_dict['points_wo_noise'] = result['points_wo_noise']
        result_dict['points_shifted_wo_noise'] = result['points_shifted_wo_noise']
        # result_dict['points_found_w_noise'] = result['points_found_w_noise']

        result_dict['ber'] = result['ber']
        result_dict['q'] = result['q']
        result_dict['evm'] = result['evm']
        result_dict['mi'] = result['mi']
        result_dict['ber_w_noise'] = result['ber_w_noise']
        result_dict['q_w_noise'] = result['q_w_noise']
        result_dict['evm_w_noise'] = result['evm_w_noise']
        result_dict['mi_w_noise'] = result['mi_w_noise']

        df = pd.concat([df, pd.DataFrame(result_dict)], ignore_index=True)

    end_time_pave = datetime.now()
    print(f'Calculation time per power: {(end_time_pave - start_time_pave).total_seconds() / 60} min')
    df.to_pickle(data_dir + 'data_collected_' + job_name + '_nch_' + str(n_channels) +
                 '_pavedbm_' + str(p_ave_dbm) + '.pkl')


end_time = datetime.now()
print(f'Calculation time: {(end_time - start_time).total_seconds() / 60} min')
print('done')