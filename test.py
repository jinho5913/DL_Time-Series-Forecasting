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
    parser = argparse.ArgumentParser(description='Test LSTM')
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
    test, scaler_y = preprocess.Scaler_test(test)

    # Build Dataset (Sliding Window)
    testX = build_dataset.build_dataset_test(np.array(test), seq_length = args.seq_length)
    print('## Shape of Data ##')
    print(testX.shape)

    # Numpy to Tensor
    testX_tensor = torch.FloatTensor(testX)

    # Define Model
    model = LSTM(args.data_dim, args.hidden_dim, args.seq_length, args.output_dim, 1).to(device)
    print(model)

    # Load Pre-trained Model
    PATH = 'pretrained_model/Sid_{}.pth'.format(sid)
    model.load_state_dict(torch.load(PATH), strict=False)

    model.eval()

    # Predict
    with torch.no_grad(): 
        pred = []
        for pr in range(len(testX_tensor)):

            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
            predicted = torch.flatten(predicted).item()
            pred.append(predicted)

        # INVERSE
        pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))

    result = [i[0] for i in pred_inverse][-1]
    print('다음날 예상 감소량 : ', result)