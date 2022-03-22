import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
import random
import argparse

from HRnet import IncUNet
from create_dataset import data_preprocess
from utils import testDataEval,save_model

def train(args):
    
    PATH = 'data'
    X_train = torch.load(f'{PATH}/ecgtoHR_train_data.pt')
    y_train = torch.load(f'{PATH}/ecgtoHR_train_labels.pt')

    X_test = torch.load(f'{PATH}/ecgtoHR_test_data.pt')
    y_test = torch.load(f'{PATH}/ecgtoHR_test_labels.pt')

    BATCH_SIZE= 64
    NUM_EPOCHS = 400
    best_loss = 1000

    train = TensorDataset(X_train,y_train)
    val = TensorDataset(X_test,y_test)
    trainLoader = DataLoader(train,batch_size = BATCH_SIZE,shuffle = True)
    valLoader = DataLoader(val, batch_size= BATCH_SIZE, shuffle=True)

    model = IncUNet((1,1,5000))
    model.cuda()
    criterion = torch.nn.SmoothL1Loss()
    optim = torch.optim.Adam(model.parameters(),lr = 0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,milestones=[100,200], gamma=0.1)

    NUM_EPOCHS = 400
    best_loss = 1000

    writer = SummaryWriter()

    if not(os.path.isdir("Saved_Model")):
        os.mkdir("Saved_Model")

    for epoch in tqdm(range(NUM_EPOCHS)):
        
        model.train()
        totalLoss = 0

        for step,(x,y) in enumerate(trainLoader):

            print('.',end = " ")
            ecg= x.unsqueeze(1).cuda()
            HR = y.unsqueeze(1).cuda()
            HR_pred = model(ecg)
            optim.zero_grad()
            loss = criterion(HR_pred,HR)
            totalLoss += loss.cpu().item()
            loss.backward()
            optim.step()

        print ('')
        print(f"Epoch:{epoch + 1} Train Loss:{totalLoss / (step+1)}")

        totalTestLoss = testDataEval(model, valLoader, criterion)
        scheduler.step()

        if best_loss > totalTestLoss:
            print ("........Saving Best Model........")
            best_loss = totalTestLoss
            save_model("Saved_Model", epoch, model, optim, best_loss )

        writer.add_scalar("Loss/test",totalTestLoss, epoch )
        writer.add_scalar("Loss/train",totalLoss/(step+1),epoch )

    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_data',action = 'store_true', help = 'Use if True')
    parser.add_argument('--sampling_freq',default = 125, type = int, help = 'Sampling Frequency')
    parser.add_argument('--upsample_freq',default = 500, type = int, help = 'Resampling Frequency')
    parser.add_argument('--window_length',default = 5,type = int,help = 'Window Length in seconds')
    parser.add_argument('--data_path',help = 'Path to dataset')
    
    parser.add_argument('--random_seed',default = 5,type = int,help = 'Random Seed initializer')
    args = parser.parse_args()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed) 

    if args.preprocess_data:
        data_preprocess(args)

    train(args)