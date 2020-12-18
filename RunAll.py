# External Imports
import pandas as pd
import torch
import torch.nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import winsound 
import time

# Local Imports
from DataParser import Pipeline
from TRN import TRN
from VisualizationHelpers import VizHelp

# Define Model Parameters
input_dim = 1
hidden_dim = 32
num_layers = 2
max_decoders = 5
num_epochs = 50
use_deltas = False
output_dim = 1

# File Locations
input_files = ["amzn.us.txt", "ibm.us.txt", "aapl.us.txt", "googl.us.txt"]

# Define a Training Model
def TrainModel(X, y, y_future, train=False, num_epochs=1):
    if train:
        print('\nBegining Training')
    else:
        print('\nBegining Testing')

    # Sound to alert user when training is complete 
    freq = 500# frequency is set to 500Hz		 
    dur = 1000# duration is set to 100 milliseconds	
    
    # Track the training errors to display later
    hist = np.zeros(num_epochs)

    with torch.set_grad_enabled(train):
        # Define the Loss Fucntion: Mean Squared Error (Note: Cross Entropy is for classification)
        criterion = torch.nn.MSELoss(reduction='mean').to(device)
        # The Optimizer defines the Method for Finding the Minima, Ex: SGD - Stochastic Gradient Descent, Adam - Adaptive Movement Estimation
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        X = X.to(device)
        y = y.to(device)
        y_future = y_future.to(device)

        for t in range(num_epochs):
            # Forward Pass: Feed Data through Model
            encoder_scores, decoder_scores = model(X)
            
            encoder_steps_taken = len(encoder_scores)
            decoder_steps_taken = len(decoder_scores)

            # Compute Encoder and Decoder Loss
            # Here we are truncating the last portion of the input data
            # because we had to cut traininng when the decoder reached the end
            # which runs "decoder_steps" ahead
            encoder_loss = criterion(encoder_scores, y[:len(encoder_scores)])
            decoder_loss = criterion(decoder_scores, y_future)

            # Compute Total Loss
            total_loss = encoder_loss + decoder_loss

            # Occasionally Print out some updates
            #if t % 5 == 0 and t != 0:
            print(f'MSE[{t}]: {total_loss.item()}')

            # Add this loss to histogram
            hist[t] = total_loss.item()

            # Only update the Model Parameters if we are in Training Mode
            if(train):
                # Zero out gradiet, or else they will accumulate between epochs
                optimiser.zero_grad()

                # Perfrom Backwards Propogation starting with the Loss, back to the source
                total_loss.backward()

                # Update parameters
                optimiser.step()
    print('Training Complete')
    winsound.Beep(freq, dur) 
    return encoder_scores, hist

# Perform Training
for decoder_steps in range(1, max_decoders+1):
    for input_file in input_files:
        # Define input file
        input_path = "input/Data/Stocks/"
        input_path += input_file

        print(f'\n\n File: {input_file}, Decoders: {decoder_steps}')

        # Create Pipeline
        pipeline = Pipeline(input_path, num_decoders=decoder_steps, use_deltas=use_deltas)

        # Instantiate the Model
        model = TRN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, num_decoders=decoder_steps)

        # Check for Optimizations
        if torch.cuda.is_available:
            print("CUDA device detected, utilizing GPU")
            device = torch.device('cuda')
        else:
            print("CUDA device detected, defaulting to CPU")
            device = torch.device('cpu')
        model.to(device)

        # Define a Training Model
        # Extract Data From Pipeline
        X_train = pipeline.X_train
        y_train = pipeline.y_train
        y_future = pipeline.D_train

        # Track Runtime
        t_start = time.perf_counter()

        # Train the Model
        y_null, errors = TrainModel(X_train, y_train, y_future, train=True, num_epochs=num_epochs)
        train_time = time.perf_counter() - t_start

        # Test the Model
        X_test = pipeline.X_test
        y_test = pipeline.y_test.to(device)
        y_pred, _ = model(X_test.to(device))

        # Evaluate the Testing
        test_criterion = torch.nn.MSELoss(reduction='mean')
        test_error = test_criterion(y_pred, y_test[:len(y_pred)])

        # Return data from GPU to CPU
        y_test_np = y_test.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()

        # Plot results and save images
        # Instantiate Visualization Helper
        viz = VizHelp()
        viz.plotPredictions(y_pred_np, y_test_np, decoder_steps, num_epochs, input_path, show=False)
        name = input_file.split('.')[0]
        viz.plotMSE(errors, decoder_steps, 1, name, show=False)

        # Append Test Error to Results File
        file1 = open("results.txt", "a")  # append mode 
        file1.write(f'{name},{decoder_steps},{test_error},{num_epochs},{train_time}\n')
        file1.close() 