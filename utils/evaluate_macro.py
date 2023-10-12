import numpy as np


def evaluate_macro(Rel, Ret):
	'''
	evaluate macro_averaged performance
	Input:
		Rel =  数据库中与测试文档类别是否相同
		Ret =  数据库的相关文档是否与测试文档在汉明距离内

	Output:
		p   = macro_averaged precision
		r   = macro_averaged recall  
	'''
	Rel_mat = np.mat(Rel)
	numTest = Rel_mat.shape[1]    #Rel_mat = size(numTrain,numTest)

	# print ('numTest=',numTest)
	precisions = np.zeros((numTest))
	recalls    = np.zeros((numTest))

	#每个元素表示对应位置的文档是否同时被认为是相关和检索到的。
	retrieved_relevant_pairs = (Rel & Ret)

	for j in range(numTest):
		#retrieved_relevant_num 表示在 retrieved_relevant_pairs 矩阵的第 j 列中，非零元素的数量。
		retrieved_relevant_num = len(retrieved_relevant_pairs[:,j][np.nonzero(retrieved_relevant_pairs[:,j])])

		#retrieved_num 表示在 Ret 矩阵的第 j 列中，非零元素的数量。
		retrieved_num = len(Ret[:, j][np.nonzero(Ret[:, j])])

		#relevant_num 表示在 Rel 矩阵的第 j 列中，非零元素的数量。
		relevant_num  = len(Rel[:, j][np.nonzero(Rel[:, j])])
		
		if retrieved_num:
			#计算了在第 j 个测试文档中的精确度，即检索到的相关文档数与检索到的文档数之比，并将结果存储在 precisions 数组的第 j 个位置。
			precisions[j] = float(retrieved_relevant_num) / retrieved_num
		
		else:
			precisions[j] = 0.0

		if relevant_num:
			#计算了在第 j 个测试文档中的召回率，即检索到的相关文档数与相关文档数之比，并将结果存储在 recalls 数组的第 j 个位置。
			recalls[j] = float(retrieved_relevant_num) / relevant_num
		
		else:
			recalls[j] = 0.0

	p = np.mean(precisions)
	r = np.mean(recalls)

	return p,r


if __name__ == '__main__':
	Rel = np.array([[True, True, False, False],[False, False, False, True]])
	Ret = np.array([[False, True, False, True],[True, True, False, False]])

	p, r = evaluate_macro(Rel, Ret)
	print('p=',p,'r=',r)