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
                temp = j
                # while temp > 0 and hammingDist[i, y[temp]] == hammingDist[i, y[temp - 1]] and new_label[0, y[temp-1]] != 1:
                #     temp -= 1
                p += float(x) / (temp + 1)
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
    traingnd = np.array([[1],[4],[2],[1], [2]])
    testgnd  = np.array([[2], [4]])
    hammingDist = np.array([[1,2],[2, 0],[1,2],[2,2], [3,2]])
    hammingDist = hammingDist.astype(float)
    # for i in range(hammingDist.shape[1]):
    #     for j in range(hammingDist.shape[0]):
    #         if traingnd[j][0] != testgnd[i][0]:
    #             hammingDist[j][i] += 0.01
    # 使用numpy的广播特性来比较traingnd和testgnd
    mask = traingnd != testgnd.T

    # 使用mask来更新hammingDist
    hammingDist[mask] += 0.01
    # hammingDist += 0.1 if np.array_equal(traingnd, testgnd) else 0
    print(hammingDist)  
    HammingRank = np.argsort(hammingDist, axis=0, )
    print(HammingRank)
    ap, p_topN = cat_apcal(traingnd, testgnd, HammingRank, 2)
    print('ap: {}, p_topN: {}'.format(ap, p_topN))