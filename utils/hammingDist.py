import numpy as np


def hammingDist(B1, B2):
	"""
	Calculate the Hamming distance between two binary arrays.

	Parameters:
		B1 (ndarray): The first binary array of shape (n1, nwords).
		B2 (ndarray): The second binary array of shape (n2, nwords).

	Returns:
		ndarray: The Hamming distance matrix of shape (n1, n2).
	"""
	#look-up table:
	bit_in_char = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 
    2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 
    3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 
    4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 
    5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 
    5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype = np.uint16)

    #compute hamming distance
	n1 = B1.shape[0]     #B1 = size(n1, nwords)
	n2, nwords = B2.shape	#B2 = size(n2, nwords)

	Dh = np.zeros((n1, n2), dtype = np.uint16)
	for i in range(n1):
		for j in range(nwords):
			y = (B1[i, j] ^ B2[:, j]).T
			Dh[i, :] = Dh[i, :] + bit_in_char[y]
	return Dh

if __name__ == '__main__':
	B1 = np.array([[1,2,3],[4,3,2]])
	B2 = np.array([[3,1,3],[3,2,1],[4,2,1]])
	# B1 = np.array([[1,0,1,1], [1,1,0,1]])
	# B2 = np.array([[0,0,1,0], [1,0,0,1], [1,1,1,1]])
	Dh = hammingDist(B1, B2)
	print (Dh)