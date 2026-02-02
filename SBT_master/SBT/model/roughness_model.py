import torch
import torch.nn as nn
import torch.nn.functional as F

class RoughnessModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 128, 32]):
        super(RoughnessModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))  # Add dropout for regularization
            prev_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.regression_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # x is the thermal embedding from the vision encoder
        return self.regression_layers(x).squeeze()  # Ensure output is [batch_size]
