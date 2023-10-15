from cmath import inf
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

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    min_loss = inf
    running_loss = running_cel_loss = running_parm_loss = running_01_loss =  0
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output, prob = model(data)
        #output = [batch, hash_num], prob = [batch, nb_classes]

        label = torch.argmax(target, dim=1)
        loss_cel = nn.CrossEntropyLoss()(prob, label)

        parameters = list(model.parameters())
        # last_layer_params = parameters[-1]
        loss_parm = sum([torch.sum(param**2) for param in parameters]) * REGULARIZER_PARAMS / 2

        loss_01 = torch.sum(torch.abs(torch.abs(output) - 1)) * LOSS_01

        loss = loss_cel

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
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH + WEIGHTS_FILE_NAME)

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)+1})]\tLoss_: {running_loss / 10:.4f},Loss_cel: {running_cel_loss / 10:.4f}, Loss_parm: {running_parm_loss / 10:.4f}, Loss_01: {running_01_loss / 10:.4f}')
                running_loss = running_cel_loss = running_parm_loss = running_01_loss =  0

    return epoch_loss / len(train_loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDH(HASH_NUM, SPLIT_NUM, 3)
    # model.load_state_dict(torch.load(WEIGHTS_SAVE_PATH + WEIGHTS_FILE_NAME))
    model = model.to(device)
    dataset = DDH_train_dataset(TRAIN_SET_PATH)
    print(len(dataset))
    # # 划分数据集
    # dataset_size = len(dataset)
    # train_size = int(0.9 * dataset_size)  # 训练集占80%
    # test_size = dataset_size - train_size  # 测试集占20%
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter()
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Train', unit='epoch'):
        print('开始训练')
        loss = train(model, train_loader, optimizer, epoch, device)
        writer.add_scalar('Loss', loss, epoch)

# 关闭SummaryWriter对象
writer.close()