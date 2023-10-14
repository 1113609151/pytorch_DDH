from model import DDH
import time
from config import *
import os
import torch
from utils.compactbit import *
from utils.hammingDist import *
from utils.evaluate_macro import *
from utils.metrics import *
from dataset import *
from torch.utils.data import DataLoader
from numpy.matlib import repmat

def model_predict(train_dataset, test_dataset, device):
    if not os.path.exists(WEIGHTS_SAVE_PATH + WEIGHTS_FILE_NAME):
        print ('no weights_file, please add weights file!')
        return
        
    print ('predict start time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    model = DDH(HASH_NUM, SPLIT_NUM, 3).to(device)
    model.load_state_dict(torch.load(WEIGHTS_SAVE_PATH + WEIGHTS_FILE_NAME))

    gallery_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    gallery_x, gallery_y = [], train_dataset.get_gallery_data()[1]
    for data, _ in gallery_loader:
        data = data.to(device)
        output, _ = model(data)
        output = output.detach().cpu().numpy()
        gallery_x.append(output)

    gallery_x = np.concatenate(gallery_x, axis=0)

    #将gallery_x中大于0的数设置为1，小于0的数设置为-1
    gallery_x[gallery_x > 0] = 1
    gallery_x[gallery_x < 0] = -1

    # print(gallery_x[0], gallery_x[1])
    print('gallery_x shape: ', gallery_x.shape)
    print('gallery_y shape: ', gallery_y.shape)
        

    query_x, query_y = [], test_dataset.get_query_data()[1]
    for data, _ in test_loader:
        data = data.to(device)
        output, _ = model(data)
        output = output.detach().cpu().numpy()
        # softmax = softmax.detach().cpu().numpy()
        # predict = np.argmax(softmax, axis=1)
        query_x.append(output)
        # predict_y.append(predict)

    query_x = np.concatenate(query_x, axis=0)
    #将query_x中大于0的数设置为1，小于0的数设置为-1
    query_x[query_x > 0] = 1
    query_x[query_x < 0] = -1

    '''
    gallery_x shape:  (63800, 48)
    gallery_y shape:  (63800,)
    query_x shape:  (7975, 48)
    query_y shape:  (7975,)
    '''

    train_binary_x, train_data_y = gallery_x, gallery_y
    train_data_y = train_data_y.reshape(-1, 1)
    test_binary_x, test_data_y = query_x, query_y
    test_data_y = test_data_y.reshape(-1, 1)
    '''
    train_binary_x shape:  (63800, 48)
    train_data_y shape:  (1,63800)
    test_binary_x shape:  (7975, 48)
    test_data_y shape:  (1,7975)
    '''

    train_y_rep = repmat(train_data_y, 1, test_data_y.shape[0])
    test_y_rep = repmat(test_data_y.T, train_data_y.shape[0], 1)
    cateTrainTest = (train_y_rep == test_y_rep)
    '''cateTrainTest shape:  (63800, 7975)'''

    train_data_y = train_data_y + 1
    test_data_y = test_data_y + 1
    train_data_y = np.asarray(train_data_y, dtype=int)
    test_data_y = np.asarray(test_data_y, dtype=int)

    B = compactbit(train_binary_x)
    tB = compactbit(test_binary_x)
    '''
    B shape:  (63800, 6)
    tB shape:  (7975, 6)
    '''

    hammRadius = 2
    hammTrainTest = hammingDist(tB, B).T
    '''hammTrainTest shape:  (7975, 63800)'''

    Ret = (hammTrainTest <= hammRadius + 0.000001)
    [Pre, Rec] = evaluate_macro(cateTrainTest, Ret)
    print ('Precision with Hamming radius_2 = ', Pre)
    print ('Recall with Hamming radius_2 = ', Rec)

    HammingRank = np.argsort(hammTrainTest, axis=0)
    [MAP, p_topN] = cat_apcal(train_data_y, test_data_y, HammingRank, TOP_K)
    print ('MAP with Hamming Ranking = ', MAP)
    print ('Precision of top %d returned = %f ' % (TOP_K, p_topN))
    print ('predict finish time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_predict(DDH_train_dataset(TRAIN_SET_PATH), DDH_test_dataset(TEST_SET_PATH), device)