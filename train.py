import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import preprocess
import build_dataset
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM')
    parser.add_argument(
        '--batch_size', type = int, default = 64)
    parser.add_argument(
        '--seq_length', type = int, default = 14)
    parser.add_argument(
        '--data_dim', type = int, default = 20)
    parser.add_argument(
        '--hidden_dim', type = int, default = 32)
    parser.add_argument(
        '--output_dim', type = int, default = 1)
    parser.add_argument(
        '--learning_rate', type = int, default = 0.01)
    parser.add_argument(
        '--n_epochs', type = int, default = 200)

    args = parser.parse_args()
    return args


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            dropout = 0.8,
                            batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias = True) 
        
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def train_model(model, train_df, num_epochs = 200, learning_rate = 0.01, verbose = 10):
     
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    n_epochs = num_epochs
    
    # epoch마다 loss 저장
    train_hist = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        avg_cost = 0
        total_batch = len(train_df)
        
        for batch_idx, samples in enumerate(train_df):

            x_train, y_train = samples

            outputs = model(x_train)
            
            loss = criterion(outputs, y_train)                    
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_cost += loss/total_batch
               
        train_hist[epoch] = avg_cost        
        
        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            
    return model.eval(), train_hist



if __name__=='__main__':
    args = parse_args()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device setting
    device = 'cpu'
    sid = str(input('device를 입력하세요 : ')) # Define Input sid
    
    # Load data
    df = preprocess.Preprocess(sid)

    # Feature Extraction
    df = preprocess.Feature_Extraction(df)

    # Generate target(weight loss)
    df = preprocess.make_target(df)
    
    # train_test_split
    train, test = preprocess.train_test_split(df)

    # Scaling(MinMax)
    train, test, scaler_y = preprocess.Scaler_train(train, test)

    # Build Dataset (Sliding Window)
    trainX, trainY = build_dataset.build_dataset_train(np.array(train), seq_length = args.seq_length)
    testX, testY = build_dataset.build_dataset_train(np.array(test),  seq_length = args.seq_length)
    print('## Shape of Data ##')
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    # Numpy to Tensor
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)
    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)

    # Define data as tensor
    dataset = TensorDataset(trainX_tensor, trainY_tensor)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,  
                            drop_last=True)

    # Define Model
    net = LSTM(args.data_dim, args.hidden_dim, args.seq_length, args.output_dim, 1).to(device)
    print(net)

    model, train_hist = train_model(net, 
                                    dataloader, 
                                    num_epochs = args.n_epochs, 
                                    learning_rate = args.learning_rate, 
                                    verbose = 20)

    PATH = 'pretrained_model/Sid_{}.pth'.format(sid)
    torch.save(model.state_dict(), PATH)

    evaluate.check_loss(model, trainX_tensor, trainY_tensor, scaler_y)