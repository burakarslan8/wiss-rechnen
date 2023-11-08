
import numpy as np
import unittest
from main import rotation_matrix, matrix_multiplication, compare_multiplication, inverse_rotation, machine_epsilon, close

class Tests(unittest.TestCase):

    def test_matrix_multiplication(self):
        a = np.random.randn(2, 2)
        c = np.random.randn(3, 3)
        self.assertTrue(np.allclose(np.dot(a, a), matrix_multiplication(a, a)))
        self.assertRaises(ValueError, matrix_multiplication, a, c)

    def test_compare_multiplication(self):
        r_dict = compare_multiplication(200, 40)
        for r in zip(r_dict["results_numpy"], r_dict["results_mat_mult"]):
            self.assertTrue(np.allclose(r[0], r[1]))

    def test_machine_epsilon(self):
        # TODO
        eps_float32 = machine_epsilon(np.float32)
        self.assertAlmostEqual(eps_float32, np.finfo(np.float32).eps, places=6, msg="Incorrect machine epsilon for float32")
        
    def test_is_close(self):
        # TODO
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.01, 2.02, 3.03])
        self.assertFalse(close(a, b, eps=0.001), msg="Matrices should not be considered close")
        
    def test_rotation_matrix(self):
        # TODO
        expected_result = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                   [np.sin(np.pi/4), np.cos(np.pi/4)]])
        result = rotation_matrix(45)
        np.testing.assert_allclose(result, expected_result, rtol=1e-10, atol=1e-10)

    def test_inverse_rotation(self):
        # TODO
        expected_result = np.array([[np.cos(np.pi/4), np.sin(np.pi/4)],
                                   [-np.sin(np.pi/4), np.cos(np.pi/4)]])
        result = inverse_rotation(45)
        np.testing.assert_allclose(result, expected_result, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
