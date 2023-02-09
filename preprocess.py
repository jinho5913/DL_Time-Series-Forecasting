import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os

def Preprocess(sid):
    ## Read Data
    df = pd.read_csv('data/data.csv')
    
    ## Basic Preprocess
    df = df[['id','date_time','sid','battery','order_weight','order_state','weight']]
    df = df.iloc[:,1:].drop_duplicates().reset_index().iloc[:,1:].rename(columns = {'index' : 'Index'})
    df['date_time'] = pd.to_datetime(df.date_time)
    df['sid'] = df.sid.astype('str')
    
    ## 무게 변동 시점별 하나의 데이터만 사용
    df_new = df.iloc[:,1:].drop(np.arange(df.shape[0]))
    df_inout = df.loc[df['sid'] == sid].sort_values('date_time')
    df_new = pd.concat([df_new, df_inout.iloc[0, :].to_frame().T], axis = 0)
    for i in tqdm(range(df_inout.shape[0])):
        try:
            before_data = df_inout.iloc[i, :]
            after_data = df_inout.iloc[i+1, :]
            if before_data.weight != after_data.weight: # 무게가 변동하는 시점(증가, 감소)일 경우
                df_new = pd.concat([df_new, after_data.to_frame().T], axis = 0)
            else:
                pass
        except:
            pass
    
    ## Reset Index
    df = df_new.reset_index().iloc[:,1:]
    
    ## Drop Outlier
    q1 = df.weight.quantile(0.25)
    q3 = df.weight.quantile(0.75)
    iqr = q3-q1

    out_idx = df['weight']>q3+1.5*iqr
    if out_idx.sum() == True:
        df.drop(out_idx, inplace = True)
        df = df.reset_index().iloc[:,1:]
    else:
        pass
    
    return df


def Feature_Extraction(df):
    #global df

    df['battery'] = df.battery.astype(float)
    df['weight'] = df.weight.astype(float)

    # Day Feature 
    df['date_time'] = df.date_time.astype(str)
    df['day'] = df.date_time.apply(lambda x : x[:10])
    df['date_time'] = pd.to_datetime(df.date_time)
    
    # Time Feature
    df['month'] = df.date_time.dt.month
    df['hour'] = df.date_time.dt.hour
    df['weekday'] = df.date_time.dt.weekday

    # Weather Feature
    weather = pd.read_csv('data/OBS_ASOS_TIM_20230201134259.csv', encoding = 'cp949')
    weather.fillna(0, inplace = True)
    weather = weather.iloc[:,2:]
    weather.columns = ['일시','기온','강수량','날씨습도','직설']
    weather['merge'] = weather.일시.apply(lambda x : x[:13])
    df['merge'] = df.date_time.astype(str).apply(lambda x : x[:13])
    df = pd.merge(df, weather, on = 'merge', how = 'inner').drop(columns = ['merge','일시'])
    df = df.sort_values('date_time').reset_index().iloc[:,1:]
    
    # Event Feature
    index_1 = df.query("'2022-04-23' >= day >= '2022-04-18'").index.tolist() + df.query("'2022-05-21' >= day >= '2022-05-16'").index.tolist()
    index_2 = df.query("'2022-04-25' >= day >= '2022-04-18'").index.tolist()
    index_3 = df.query("'2022-07-12' >= day >= '2022-06-13'").index.tolist()
    df['event'] = 0
    for i in index_1:
        df.loc[i, 'event'] = 6.11
    for i in index_2:
        df.loc[i, 'event'] = 0.18
    for i in index_3:
        df.loc[i, 'event'] = 0.0917
        
    # Battery_weight Feature
    df = pd.merge(df, df.groupby('battery')['weight'].mean().reset_index().rename(columns = {'weight' : 'battery_weight'}), on = 'battery')
    df = df.sort_values('date_time').reset_index().iloc[:,1:]
    
    # Weight_mean Feature
    df = pd.merge(df, df.groupby('month')['weight'].mean().reset_index().rename(columns = {'weight' : 'month_weight'}), on = 'month') # 월 별 weight 변화량
    df = pd.merge(df, df.groupby('hour')['weight'].mean().reset_index().rename(columns = {'weight' : 'hour_weight'}), on = 'hour') # 시간 별 weight 변화량
    df = pd.merge(df, df.groupby('weekday')['weight'].mean().reset_index().rename(columns = {'weight' : 'weekday_weight'}), on = 'weekday') # 요일 별 weight 변화량
    df = df.sort_values('date_time').reset_index().iloc[:,1:]
    
    # Battery_mean Feature
    df = pd.merge(df, df.groupby('month')['battery'].mean().reset_index().rename(columns = {'battery' : 'month_battery'}), on = 'month') # 월 별 battery 변화량
    df = pd.merge(df, df.groupby('hour')['battery'].mean().reset_index().rename(columns = {'battery' : 'hour_battery'}), on = 'hour') # 시간 별 battery 변화량
    df = pd.merge(df, df.groupby('weekday')['battery'].mean().reset_index().rename(columns = {'battery' : 'weekday_battery'}), on = 'weekday') # 요일 별 battery 변화량
    df = df.sort_values('date_time').reset_index().iloc[:,1:]
    
    # Weight_ratio Feature
    df_real = pd.DataFrame({'weight_ratio':[]})
    lst = [0]
    df_new = df.sort_values('date_time').reset_index().iloc[:,1:]
    for i in range(df_new.shape[0]-1):
        before = df_new.iloc[i,:].weight # 이전 무게
        after = df_new.iloc[i+1,:].weight # 현재 무게
        if before - after > 0: 
            decrease = round(before - after, 2) # 이전보다 현재 무게가 감소하였음
            lst.append(decrease * (-1))
        elif before - after < 0:
            increase = round(after - before, 2)
            lst.append(increase)
        else:
            lst.append(0)
    target = pd.concat([df_new.reset_index().iloc[:,1:], pd.DataFrame({'weight_ratio' : lst})], axis = 1)
    df_real = pd.concat([df_real, target], axis = 0)
    df = df_real.sort_values('date_time').reset_index().iloc[:,1:]
    
    # Duration
    h_lst = [0]
    for i in range(df_new.shape[0]-1):
        before_data = df_new.iloc[i, :]
        after_data = df_new.iloc[i+1, :]
        if before_data.weight > after_data.weight: #무게 감소가 측정되면
            hours = (after_data.date_time - before_data.date_time).seconds/3600 # 출입 - 출고 가 얼마나 걸렸는지 'hour' 파악
            h_lst.append(hours)
        else:
            h_lst.append(0)
    df = pd.concat([df, pd.DataFrame({'Duration' : h_lst})], axis = 1)
    df = df.sort_values('date_time')

    return df


def make_target(df):
    df_target = pd.DataFrame({'target':[]})
    lst = [0]
    df_n = df.sort_values('date_time')
    for i in range(df_n.shape[0]-1):
        before = df_n.iloc[i,:].weight # 이전 무게
        after = df_n.iloc[i+1,:].weight # 현재 무게
        if before - after > 0: 
            decrease = round(before - after, 2) # 이전보다 현재 무게가 감소하였음
        else:
            decrease = 0 # 유지, 증가의 경우는 모두 0으로 할당
        lst.append(decrease)
    target = pd.concat([df_n.reset_index().iloc[:,1:], pd.DataFrame({'target' : lst})], axis = 1)
    df_target = pd.concat([df_target, target], axis = 0)
    
    df_target = df_target[['weight_ratio', 'battery', 'hour', 'month', 'weekday','Duration', 'event', '기온', '강수량', '날씨습도', '직설', 'battery_weight','month_weight', 'hour_weight', 'weekday_weight', 'month_battery','hour_battery', 'weekday_battery', 'weight','target']]
    
    return df_target

def train_test_split(df):
    train_size = int(len(df)*0.8)
    train = df[0:train_size].reset_index().iloc[:,1:]
    test = df[train_size:].reset_index().iloc[:,1:]
    
    return train, test


def Scaler_train(train, test):        
    # X_scale
    scaler_x = MinMaxScaler()
    scaler_x.fit(train.iloc[:,:-1])
    train.iloc[:, :-1] = scaler_x.transform(train.iloc[:, :-1])
    test.iloc[:, :-1] = scaler_x.transform(test.iloc[:, :-1])

    # Y_scale
    scaler_y = MinMaxScaler()
    scaler_y.fit(train.iloc[:, [-1]])
    train.iloc[:, [-1]] = scaler_y.transform(train.iloc[:, [-1]])
    test.iloc[:, [-1]] = scaler_y.transform(test.iloc[:, [-1]])

    train = train.values
    test = test.values
        
    return train, test, scaler_y


def Scaler_test(df):
    scaler_x = MinMaxScaler()
    scaler_x.fit(df.iloc[:,:-1])
    df.iloc[:, :-1] = scaler_x.transform(df.iloc[:, :-1])
        
    scaler_y = MinMaxScaler()
    scaler_y.fit(df.iloc[:, [-1]])
    df.iloc[:, [-1]] = scaler_y.transform(df.iloc[:, [-1]])
        
    test = df.values
        
    return test,  scaler_y