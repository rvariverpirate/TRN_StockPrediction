###################################
# TRN Modified for Financial Data #
###################################
import torch
import torch.nn as nn
import numpy as np

def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(inplace=inplace))


class TRN(nn.Module): #  Inherrits from nn.Module
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=2, num_decoders=3):
        super(TRN, self).__init__()
        
        # Assign model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_decoders = num_decoders
        self.future_dim = input_dim# Future size same as current size
        self.fusion_dim = input_dim*2# Both current and future states
        
        # Define Transforms
        self.hx_trans = fc_relu(self.hidden_dim, self.hidden_dim) # Hidden State Transform, Encoder to Decoder
        self.cx_trans = fc_relu(self.hidden_dim, self.hidden_dim) # Cell State Trandform, Encoder to Decoder 
        self.fusion_linear = fc_relu(self.output_dim, self.hidden_dim) # Pass Predictions from one Decoder to Next
        self.future_linear = fc_relu(self.hidden_dim, self.future_dim) # Pass Hidden Layers from all Decoders to Future Vector
        
        # Define Encoder Cell
        self.enc_cell = nn.LSTMCell(self.fusion_dim, hidden_dim, num_layers)
        
        # Define Decoder Cell
        # TODO: Consider adding dropout
        self.dec_cell = nn.LSTMCell(hidden_dim, hidden_dim, num_layers)
        
        # Define the Classifier (Readout Layer)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    # Define the Encoder
    def encoder(self, current_input, future_input, enc_hx, enc_cx):        
        # Combine Current and Future Predicted Values
        fusion_input = torch.cat((current_input, future_input), 0).view(1, -1)
        enc_hx, enc_cx = self.enc_cell(fusion_input, (enc_hx, enc_cx))
        # TODO: consider applying dropout
        enc_score = self.classifier(enc_hx)
        return enc_hx, enc_cx, enc_score
    
    # Define the Decoder
    def decoder(self, fusion_input, dec_hx, dec_cx):
        
        dec_hx, decc_cx = self.dec_cell(fusion_input, (dec_hx, dec_cx))
        dec_score = self.classifier(dec_hx)
        return dec_hx, dec_cx, dec_score
    
    def forward(self, X):
        batch_size = X.shape[0]
        enc_hx = X.new_zeros((1, self.hidden_dim))
        enc_cx = X.new_zeros((1, self.hidden_dim))
        future_input = X.new_zeros((1, self.future_dim))
        
        # Track how well the model is doing
        encoder_scores = []
        decoder_scores = []
        
        # Encoder
        for enc_step in range(batch_size-self.num_decoders):
            
            # Encoder: Pass Data at time t through encder
            enc_hx, enc_cx, enc_score = self.encoder(X[enc_step,:], future_input[0,:], enc_hx, enc_cx)
            encoder_scores.append(enc_score)
            
            # Decoder: 
            dec_hx = self.hx_trans(enc_hx)
            dec_cx = self.cx_trans(enc_cx)

            # Zero out inputs for first decoder stage
            fusion_input = X.new_zeros((1, self.hidden_dim))
            future_input = X.new_zeros((1, self.future_dim))
            
            # Decoder: results through serires of decoder steps
            for dec_step in range(self.num_decoders):
                dec_hx, dec_cx, dec_score = self.decoder(fusion_input, dec_hx, dec_cx)
                decoder_scores.append(dec_score)
                # Update the input for the next decoder
                fusion_input = self.fusion_linear(dec_score)
                
                # Sum the predicted future values from each decoder step
                future_input = future_input + self.future_linear(dec_hx)
            # Tave average of future input sum
            future_input = future_input/self.num_decoders
            
        encoder_scores = torch.stack(encoder_scores, dim=1).view(-1, self.output_dim)
        decoder_scores = torch.stack(decoder_scores, dim=1).view(-1, self.output_dim)
        return encoder_scores, decoder_scores