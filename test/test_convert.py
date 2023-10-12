import numpy as np
import unittest

def convert(arr):
    """
    Converts an array of binary digits to an integer value.

    Parameters:
    - arr (List[int]): The array of binary digits to convert.

    Returns:
    - int: The integer value converted from the binary array.
    """
    arr_mat = np.mat(arr)
    [_, col] = arr_mat.shape
    value = 0
    for i in range(col):
        value = value + (2 ** i) * arr[i]
    
    return value

class ConvertTest(unittest.TestCase):
    def test_convert_empty_array(self):
        self.assertEqual(convert([]), 0)
    
    def test_convert_single_digit(self):
        self.assertEqual(convert([0]), 0)
        self.assertEqual(convert([1]), 1)
    
    def test_convert_multiple_digits(self):
        self.assertEqual(convert([0, 0, 0]), 0)
        self.assertEqual(convert([1, 0, 0]), 1)
        self.assertEqual(convert([1, 1, 0]), 3)
        self.assertEqual(convert([1, 1, 1]), 7)

if __name__ == '__main__':
    unittest.main()