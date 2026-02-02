import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F

class BCModel(nn.Module):
    def __init__(self, latent_size=2048, hidden_layers=[1024, 512, 128, 64], output_dim=2, dropout_rate=0.5):
        super().__init__()
        
        # Build a deeper MLP with more capacity
        layers = []
        
        # Input layer
        layers.append(nn.Linear(latent_size, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Optional: Initialize weights with a custom strategy
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, thermal_embeddings):
        # Process the thermal embeddings through the MLP
        cmd_vel_pred = self.mlp(thermal_embeddings)
        return cmd_vel_pred

