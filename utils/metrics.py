import numpy as np

def cat_apcal(gallery_set_y: np.ndarray, query_set_y: np.ndarray, HammingRank: np.ndarray, top_N: int) -> tuple[float, float]:    
    # 获取训练样本和测试样本的数量
    numgallery, numquery = HammingRank.shape
    
    # 计算每个测试样本的AP
    ap_all = np.zeros(numquery)
    for i in range(numquery):
        y = HammingRank[:, i]
        x = 0
        p = 0
        new_label = np.zeros((1, numgallery))
        new_label[gallery_set_y.T == query_set_y[i]] = 1
        num_return_NN = numgallery
        for j in range(num_return_NN):
            if new_label[0, y[j]] == 1:
                x += 1
                p += float(x) / (j + 1)
        if p == 0:
            ap_all[i] = 0
        else:
            ap_all[i] = p / x
    
    # 计算每个测试样本的top N精度
    pall = np.zeros(numquery)
    for i in range(numquery):
        y_1 = HammingRank[:, i]
        n = 0
        new_label_1 = np.zeros((1, numgallery))
        new_label_1[gallery_set_y.T == query_set_y[i]] = 1
        for jj in range(top_N):
            if new_label_1[0, y_1[jj]] == 1:
                n = n + 1
        pall[i] = 1.0 * n / top_N
    
    # 计算平均精度（AP）和top N精度
    ap = np.mean(ap_all)
    p_topN = np.mean(pall)
    
    return ap, p_topN


if __name__ == '__main__':
    traingnd = np.array([[1],[4],[2],[1]])
    testgnd  = np.array([[2],[4]])
    hammingDist = np.array([[1,2],[2,1],[3,1],[0,2]])
    HammingRank = np.argsort(hammingDist, axis=0)
    ap, p_topN = cat_apcal(traingnd, testgnd, HammingRank, 2)
    print('ap: {}, p_topN: {}'.format(ap, p_topN))