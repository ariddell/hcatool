"""
Tests for `hcatool` module.
"""
import os

import numpy as np

import hcatool
import hcatool.tests.base as base


class TestHcatool(base.TestCase):

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        self.datastem = os.path.join(test_dir, 'reuters')
        self.fitstem = os.path.join(test_dir, 'fit-reuters-LDA-K20-seed5-iter100')
        super().setUp()

    def test_hcatool_load_counts(self):
        datastem, fitstem = self.datastem, self.fitstem
        ndt_df, ntw_df = hcatool.load_counts(datastem, fitstem)
        np.testing.assert_array_equal(ndt_df >= 0, True)
        np.testing.assert_array_equal(ntw_df >= 0, True)

    def test_hcatool_load(self):
        datastem, fitstem = self.datastem, self.fitstem
        theta_df, phi_df = hcatool.load(datastem, fitstem)
        np.testing.assert_allclose(phi_df.sum(axis=1).values, 1)
        np.testing.assert_allclose(theta_df.sum(axis=1).values, 1)
