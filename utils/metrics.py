import numpy as np
import torch
def cat_apcal(gallery_set_y: torch.tensor, query_set_y: torch.tensor, HammingRank: torch.tensor, top_N: int):  

    numgallery, numquery = HammingRank.shape
    ap_all = torch.zeros(numquery)
    
    for i in range(numquery):
        y = HammingRank[:, i]
        x = 0
        p = 0
        new_label = (gallery_set_y.T == query_set_y[i]).float()
        num_return_NN = numgallery
        
        mask = new_label[:, y].eq(1)
        cum_sum = torch.cumsum(mask, dim=1)
        denom = torch.arange(1, num_return_NN + 1, device=HammingRank.device).view(1, -1)
        precision = torch.where(mask, cum_sum / denom, torch.zeros_like(cum_sum))
        p = torch.sum(precision)
        x = torch.sum(mask)
        
        if x == 0:
            ap_all[i] = 0
        else:
            ap_all[i] = p / x
    
    ap = torch.mean(ap_all).item()
    
    return ap, 0

# def cat_apcal(gallery_set_y: torch.tensor, query_set_y: torch.tensor, HammingRank: torch.tensor, top_N: int):  

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # gallery_set_y = gallery_set_y.to(device)
#     # query_set_y = query_set_y.to(device)
#     # HammingRank = HammingRank.to(device)

#     # # 获取训练样本和测试样本的数量
#     # numgallery, numquery = HammingRank.shape
    
#     # # 计算每个测试样本的AP
#     # new_label = torch.zeros((numgallery, numquery)).to(device)
#     # new_label[gallery_set_y.T.unsqueeze(1) == query_set_y.unsqueeze(0)] = 1
    
#     # x = torch.cumsum(new_label[HammingRank], dim=1)
#     # j = torch.arange(1, numgallery+1).unsqueeze(0).to(device)
#     # p = torch.sum(x / j, dim=1)
    
#     # ap_all = p / torch.sum(new_label, dim=1)
#     # ap_all[torch.isnan(ap_all)] = 0
    
#     # # 计算平均精度（AP）和top N精度
#     # ap = torch.mean(ap_all)
    
#     # return ap, 0




#     # 获取训练样本和测试样本的数量
#     numgallery, numquery = HammingRank.shape
#     # print(gallery_set_y.is_cuda(), query_set_y.is_cuda())
#     # 计算每个测试样本的AP
#     # ap_all = np.zeros(numquery)
#     ap_all = torch.zeros(numquery)
#     for i in range(numquery):
#         y = HammingRank[:, i]
#         x = 0
#         p = 0
#         # new_label = np.zeros((1, numgallery))
#         new_label = torch.zeros((1, numgallery))
#         new_label[gallery_set_y.T == query_set_y[i]] = 1
#         num_return_NN = numgallery
#         for j in range(num_return_NN):
#             if new_label[0, y[j]] == 1:
#                 x += 1
#                 p += float(x) / (j + 1)
#         if p == 0:
#             ap_all[i] = 0
#         else:
#             ap_all[i] = p / x
    
#     # # 计算每个测试样本的top N精度
#     # pall = np.zeros(numquery)
#     # for i in range(numquery):
#     #     y_1 = HammingRank[:, i]
#     #     n = 0
#     #     new_label_1 = np.zeros((1, numgallery))
#     #     new_label_1[gallery_set_y.T == query_set_y[i]] = 1
#     #     for jj in range(top_N):
#     #         if new_label_1[0, y_1[jj]] == 1:
#     #             n = n + 1
#     #     pall[i] = 1.0 * n / top_N
    
#     # 计算平均精度（AP）和top N精度
#     # ap = np.mean(ap_all)
#     ap = torch.mean(ap_all)
#     # p_topN = np.mean(pall)
    
#     return ap, 0
    # return ap, p_topN

# def cat_apcal(gallery_set_y: np.ndarray, query_set_y: np.ndarray, HammingRank: np.ndarray, top_N: int):
#     # 计算每个测试样本的AP
#     numgallery, numquery = HammingRank.shape
#     new_label = np.equal(gallery_set_y.T, query_set_y[:, np.newaxis])
#     x = np.cumsum(new_label, axis=1)
#     precision = np.divide(x, np.arange(1, numgallery + 1))
#     recall = np.divide(x, np.sum(new_label, axis=1, keepdims=True))
#     ap_all = np.mean(np.where(new_label, precision, 0), axis=1)

#     # 计算每个测试样本的top N精度
#     top_N_predictions = HammingRank[:, :top_N]
#     top_N_labels = new_label[:, top_N_predictions]
#     p_topN = np.mean(np.any(top_N_labels, axis=1))

#     return np.mean(ap_all), p_topN


if __name__ == '__main__':
    traingnd = np.array([[1],[4],[2],[1]])
    testgnd  = np.array([[2],[4]])
    hammingDist = np.array([[1,2],[2,1],[3,1],[0,2]])
    HammingRank = np.argsort(hammingDist, axis=0)
    ap, p_topN = cat_apcal(traingnd, testgnd, HammingRank, 2)
    print('ap: {}, p_topN: {}'.format(ap, p_topN))


