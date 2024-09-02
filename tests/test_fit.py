import unittest
import pandas as pd
import numpy as np

from triplet_fitting.fit import fit_all, fit_triplets
from triplet_fitting.dataset import form_datasets_triplets
from triplet_fitting.matlab_interface import start_matlab_engine


class TestConversionTripletFitting(unittest.TestCase):

    def setUp(self):
        # Setup any initial data or configurations needed for the tests
        self.data_dir = "path/to/data_dir"
        self.filename_prefix = "test_data"
        self.p_ave_dbm_list = [10.0, 12.0]
        self.n_runs = 10
        self.H_max = 2
        self.channel_noise = False
        self.independant_gauss = False

        # Example DataFrame setup for testing
        self.df = pd.DataFrame({
            'run': np.tile(np.arange(10), 10),
            'p_ave_dbm': np.repeat([10.0, 12.0], 50),
            'scale_coef': np.random.random(100),
            'points_shifted': [np.random.random(10) + 1j * np.random.random(10) for _ in range(100)],
            'points_shifted_wo_noise': [np.random.random(10) + 1j * np.random.random(10) for _ in range(100)],
            'points_orig': [np.random.random(10) + 1j * np.random.random(10) for _ in range(100)],
            'orig_m1_real': np.random.random(100),
            'orig_m1_imag': np.random.random(100),
            'orig_real': np.random.random(100),
            'orig_imag': np.random.random(100),
            'orig_p1_real': np.random.random(100),
            'orig_p1_imag': np.random.random(100),
            'shifted_real': np.random.random(100),
            'shifted_imag': np.random.random(100)
        })

    def test_form_datasets_triplets(self):
        # Test the form_datasets_triplets function
        runs = [0, 1, 2]
        p_ave_dbm = 10.0
        M_shifted = 1
        M_orig = 1

        dataset, target_dataset = form_datasets_triplets(self.df, runs, p_ave_dbm, M_shifted, M_orig,
                                                         self.channel_noise)

        self.assertEqual(dataset.shape[0], target_dataset.shape[0])
        self.assertTrue(dataset.shape[1] > 0)
        self.assertTrue(target_dataset.shape[1] > 0)

    def test_fit_triplets(self):
        # Test the fit_triplets function
        eng = start_matlab_engine()
        runs = [0, 1, 2]
        p_ave_dbm = 10.0
        M_shifted = 1
        M_orig = 1

        dataset, target_dataset = form_datasets_triplets(self.df, runs, p_ave_dbm, M_shifted, M_orig,
                                                         self.channel_noise)
        df_new = pd.concat([pd.DataFrame(dataset), pd.DataFrame(target_dataset)], axis=1).astype(float)

        df_new.columns = [f"shifted_{i}" for i in range(-M_shifted, M_shifted + 1)] + \
                         [f"orig_{i}" for i in range(-M_orig, M_orig + 1)]

        df_result = fit_triplets(df_new, path_to_save=f"{self.data_dir}/test_triplets_fit.pkl",
                                 H_max=self.H_max, matlab_engine=eng, disp=False,
                                 independant_gauss=self.independant_gauss)

        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertTrue(df_result.shape[0] > 0)

    def test_fit_all(self):
        # Test the fit_all function
        df_result = fit_all(self.p_ave_dbm_list, self.data_dir, self.filename_prefix,
                            n_runs=self.n_runs, H_max=self.H_max,
                            independant_gauss=self.independant_gauss,
                            channel_noise=self.channel_noise)

        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertTrue(df_result.shape[0] > 0)


if __name__ == '__main__':
    unittest.main()
