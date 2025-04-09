import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            # Convolutional layers
            # This layer takes a 1-channel input (grayscale image) and applies a 10x10 kernel and converts it to 64 channels
            # Output size is 96x96
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(),
            # Ouput size is 48x48
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=7),  # -> 42x42
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 21x21

            nn.Conv2d(128, 128, kernel_size=4), # -> 18x18
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 9x9

            nn.Conv2d(128, 256, kernel_size=4), # -> 6x6
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Euclidean distance (L1 or L2 both can work; L1 in paper)
        return F.pairwise_distance(output1, output2)
        
