import numpy as np
import torch

def MAE(true, pred):
    return np.mean(np.abs(true-pred))


def check_loss(model, X, target, scaler):
    '''
    X : train.py = trainX_tensor

    target : train.py = trainY_tensor

    scaler : to inverse predict values
    '''

    with torch.no_grad():
        pred = []
        for pr in range(len(X)):
            model.reset_hidden_state()
            predicted = model(torch.unsqueeze(X[pr], 0))
            predicted = torch.flatten(predicted).item()
            pred.append(predicted)
        
        pred_inverse = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
        trainY_inverse = scaler.inverse_transform(target)

    print('##MAE SOCRE :',  MAE(pred_inverse, trainY_inverse, '##'))