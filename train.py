from cmath import inf
import time
from tqdm import tqdm
from config import *
from model import DDH
from dataset import *
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import logging
from predict import model_predict

def getLogger(text):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                datefmt="%a %b %d %H:%M:%S %Y")
    
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join(TRAIN_LOG,
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/' + text + '_' +'log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger


def train(model, train_loader, optimizer, epoch, device, creiation, logger):
    global max_acc
    global max_map

    print('进入训练阶段')
    model.train()
    min_loss = inf
    running_loss = running_cel_loss = running_parm_loss = running_01_loss =  0
    epoch_loss = 0
    MAP = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        output, prob = model(data)
        #output = [batch, hash_num], prob = [batch, nb_classes]

        label = torch.argmax(label, dim=1)
        loss_cel = creiation(prob, label)

        parameters = list(model.parameters())
        # last_layer_params = parameters[-1]
        loss_parm = sum([torch.sum(param**2) for param in parameters]) * REGULARIZER_PARAMS / 2

        loss_01 = torch.sum(torch.abs(torch.abs(output) - 1)) * LOSS_01

        loss = loss_cel + loss_01 + loss_parm

        running_loss += loss
        running_cel_loss += loss_cel
        running_parm_loss += loss_parm
        running_01_loss += loss_01

        epoch_loss += loss

        if min_loss > loss:
            min_loss = loss
            #检查是否存在WEIGHTS_SAVE_PATH，没有的话创建文件夹
            if not os.path.exists(WEIGHTS_SAVE_PATH):
                os.makedirs(WEIGHTS_SAVE_PATH)
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH + f'{HASH_NUM}'+'_'+f'{LOSS_01}'+ WEIGHTS_FILE_NAME)

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
                # print(f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)+1})]\tLoss_: {running_loss / 10:.4f},Loss_cel: {running_cel_loss / 10:.4f}, Loss_parm: {running_parm_loss / 10:.4f}, Loss_01: {running_01_loss / 10:.4f}')
                logger.info(f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)+1})]\tLoss_: {running_loss / 10:.4f},Loss_cel: {running_cel_loss / 10:.4f}, Loss_parm: {running_parm_loss / 10:.4f}, Loss_01: {running_01_loss / 10:.4f}')
                running_loss = running_cel_loss = running_parm_loss = running_01_loss =  0

    with torch.no_grad():
        print("进入测试阶段")
        model.eval()
        all_acc = 0
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            label = torch.argmax(label, dim=1)
            output, prob = model(data)
            pred = torch.argmax(prob, dim=1)
            acc = torch.sum(pred == label) / len(label)
            all_acc += acc
        
        all_acc = all_acc / len(test_loader)
        if max_acc < all_acc:
            max_acc = all_acc
        #     if not os.path.exists(WEIGHTS_SAVE_PATH):
        #         os.makedirs(WEIGHTS_SAVE_PATH)
        #     torch.save(model.state_dict(), WEIGHTS_SAVE_PATH + f'{HASH_NUM}'+'_'+f'{LOSS_01}'+ WEIGHTS_FILE_NAME)
        
        print(f'Eval Epoch: {epoch} \tacc: {all_acc:.4f}  max_acc: {max_acc:.4f}')

        if epoch > 200:
            MAP, _ = model_predict(train_dataset, test_dataset, device, model)
            if max_map < MAP:
                max_map = MAP
                if not os.path.exists(WEIGHTS_SAVE_PATH):
                    os.makedirs(WEIGHTS_SAVE_PATH)
                torch.save(model.state_dict(), WEIGHTS_SAVE_PATH + f'{HASH_NUM}'+'_'+f'{LOSS_01}'+ '_'+'MAX_MAP '+ WEIGHTS_FILE_NAME)

        print(f'Eval Epoch: {epoch} \tMAP: {MAP:.4f}  max_map: {max_map:.4f}')

    return epoch_loss / len(train_loader), all_acc, MAP


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDH(HASH_NUM, SPLIT_NUM, 3)
    # model.load_state_dict(torch.load(WEIGHTS_SAVE_PATH + f'{HASH_NUM}'+'_'+f'{LOSS_01}'+ WEIGHTS_FILE_NAME))
    model = model.to(device)
    dataset = DDH_train_dataset(TRAIN_SET_PATH)
    print(f'dataset size:{len(dataset)}')
    
    # # 划分数据集
    train_dataset, test_dataset = DDH_train_dataset(TRAIN_SET_PATH), DDH_test_dataset(TEST_SET_PATH)
    print(f'train_dataset size:{len(train_dataset)}, test_dataset size:{len(test_dataset)}')


    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)
    writer = SummaryWriter()
    crieation = nn.CrossEntropyLoss()
    text = f'batch_size: {BATCH_SIZE}, lr: {LR}, hash_num: {HASH_NUM}, LOSS_01: {LOSS_01}'
    logger = getLogger(text)
    logger.info(f'batch_size: {BATCH_SIZE}, lr: {LR}, hash_num: {HASH_NUM}, LOSS_01: {LOSS_01}, REGULARIZER_PARAMS: {REGULARIZER_PARAMS}')
    max_acc = 0
    max_map = 0
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Train', unit='epoch'):
        print('开始训练')
        loss, acc, MAP = train(model, train_loader, optimizer, epoch, device, crieation, logger)
        scheduler.step()
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('acc', acc, epoch)

# 关闭SummaryWriter对象
writer.close()