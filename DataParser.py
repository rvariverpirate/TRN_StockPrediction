import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Check Cuda Version: nvcc --version
class Pipeline():
    def __init__(self, file_path, parse_dates=True, start_end=('2010-01-02', '2017-10-11'), column_name='Close', num_decoders=3, use_deltas=True, expand=True, test_ratio=0.2):
        dates = pd.date_range(start_end[0], start_end[1],freq='B') # Used to establish a Date Range of interest
        dates_df = pd.DataFrame(index=dates)
        raw_data = pd.read_csv(file_path, parse_dates=parse_dates, index_col=0)
        raw_data=dates_df.join(raw_data)# Selects on the data within the range        
        
        self.column_name = column_name
        self.num_decoders=num_decoders

        # Sanitize Data
        # Replace any missing data. ffill: propagate last valid observation forward to next valid
        selected_data = raw_data[[column_name]].fillna(method='ffill')

        # Scale data between -1 and 1 using  SciKit's MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))# Should we really do this, couldn't magnitude be useful?
        # Also any stock that goes negtive I'm buying
        selected_data[column_name] = scaler.fit_transform(selected_data[column_name].values.reshape(-1,1))
        
        # Make dataframe a class attribute
        self.data_df = selected_data

        # Make a Target Class (This can optionally be modified)
        if use_deltas:
            print('Using Deltas:')
            self.convertToDelta()
            self.posNegSplit()

        else:
            print('Using Raw Input:')
            self.split()

        self.ourputs = (self.X_train, self.X_test, self.y_train, self.y_test)

    def convertToDelta(self):
        # .pct_change()? Not quite right
        # Note: When computing difference, first element is undefined
        self.target = self.data_df.diff().iloc[1:].values.flatten()
        # Ater splicing first NAN element the 
        # slope of current point is based on difference of
        # the current and the next point, which we want
        # But still need to make the frames the same size,
        # so delete last element of input
        self.data_df = self.data_df.iloc[:-1]
        self.data = self.data_df.values.flatten()

    def posNegSplit(self):
        self.dy_pos = (0.5 + self.target/2.0).flatten()
        self.dy_neg = (0.5 - self.target/2.0).flatten()
        self.dy_pos_neg = np.array([self.dy_pos, self.dy_neg]).T
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.dy_pos_neg, test_size=0.2, random_state=42, shuffle=False)
        
        # Expand the Data for the Decoder (Use Future values i.e. target y)
        self.D_train = self.expand_data(y_train, 1).view(-1, 2)
        self.D_test = self.expand_data(y_test, 1).view(-1, 2)
        
        # Convert Remaining data to Pytorch Tensors (expand method does this automatically)
        self.X_train = torch.from_numpy(X_train).type(torch.Tensor).view(-1, 1)#.view(1, 1, -1))
        self.X_test = torch.from_numpy(X_test).type(torch.Tensor).view(-1, 1)
        self.y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1, 2)
        self.y_test = torch.from_numpy(y_test).type(torch.Tensor).view(-1, 2)

    def split(self):
        # Target should be tommorows stock
        self.data = self.data_df.values.flatten()
        # Shift target left by one to become future
        self.target = self.data[1:]
        # Remove last data point because no future value is known for it
        self.data = self.data[:-1]
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42, shuffle=False)
        
        # Expand the Data for the Decoder (Use Future values i.e. target y)
        self.D_train = self.expand_data(y_train, 1).view(-1, 1)
        self.D_test = self.expand_data(y_test, 1).view(-1, 1)
        
        # Convert Remaining data to Pytorch Tensors (expand method does this automatically)
        self.X_train = torch.from_numpy(X_train).type(torch.Tensor).view(-1, 1)#.view(1, 1, -1))
        self.X_test = torch.from_numpy(X_test).type(torch.Tensor).view(-1, 1)
        self.y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1, 1)
        self.y_test = torch.from_numpy(y_test).type(torch.Tensor).view(-1, 1)

    # Expand the full data set for 
    def expand_data(self, data, input_dim):
        result = []
        for i in range(len(data)-self.num_decoders):
            result += list(data[i:i+self.num_decoders])
        return torch.from_numpy(np.asarray(result)).type(torch.Tensor)