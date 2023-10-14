import numpy as np


import numpy as np

def compactbit(binary_matrix):
    """
    Converts a binary matrix into a compact bit representation.

    Parameters:
    - binary_matrix (ndarray): The binary matrix to be converted.

    Returns:
    - compact_bit (ndarray): The compact bit representation of the binary matrix.
    """
    num_samples, num_bits = binary_matrix.shape
    num_words = int(np.ceil(num_bits / 8))
    compact_bit = np.zeros((num_samples, num_words), dtype=np.uint8)

    for i in range(num_samples):
        for j in range(num_words):
            temp = binary_matrix[i, j * 8 : (j + 1) * 8]
            value = convert(temp)
            compact_bit[i, j] = value

    return compact_bit

def convert(arr):
    """
    Converts an array of binary digits to an integer value.

    Parameters:
    - arr (List[int]): The array of binary digits to convert.

    Returns:
    - int: The integer value converted from the binary array.
    """
    value = 0
    for i, digit in enumerate(arr):
        value += (2 ** i) * digit
    
    return value


if __name__ == '__main__':
	b = np.array([[0,0,1,1,0,0,0,1,1,1],[0,0,0,1,0,0,1,1,1,1]])
	#测试convert
	print(convert([1,-1,1,1]))

	#测试compactbit
	# print(compactbit(b))
