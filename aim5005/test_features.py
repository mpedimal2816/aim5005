from aim5005.features import MinMaxScaler, StandardScaler
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)  # Added this line to define result
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))


    def test_standard_scaler_zero_variance(self):
        scaler = StandardScaler()
        data = np.array([[4, 0], [4, 0], [4, 0], [4, 0]])
        scaler.fit(data)
        expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])  # Expect 0s since the mean should be subtracted and divided by the std (which is zero)
        result = scaler.transform(data)
        assert (result[:, 1] == expected[:, 1]).all(), "Scaler does not handle zero variance correctly. Expect {}. Got: {}".format(expected[:, 1], result[:, 1])


    class TestLabelEncoder:
        def test_fit(self):
            le = LabelEncoder()
            le.fit(["paris", "paris", "tokyo", "amsterdam"])
            assert np.array_equal(le.classes_, ["amsterdam", "paris", "tokyo"])

        def test_transform(self):
            le = LabelEncoder()
            le.fit(["paris", "tokyo", "amsterdam"])
            transformed = le.transform(["tokyo", "tokyo", "paris"])
            assert np.array_equal(transformed, [2, 2, 1])

        def test_fit_transform(self):
            le = LabelEncoder()
            transformed = le.fit_transform(["paris", "tokyo", "amsterdam", "tokyo"])
            assert np.array_equal(transformed, [1, 2, 0, 2])

        def test_transform_unseen(self):
            le = LabelEncoder()
            le.fit(["paris", "tokyo"])
            with pytest.raises(ValueError):
                le.transform(["london"])

if __name__ == '__main__':
    unittest.main()