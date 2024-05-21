import torch
import torch.nn as nn
    
class SpatialStreamConvNet(nn.Module):
    def __init__(self, num_classes=101):
        super(SpatialStreamConvNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3,96, kernel_size=7, stride=2),
            nn.LocalResponseNorm(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(96,256, kernel_size=5, stride=2),
            nn.LocalResponseNorm(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256,512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512,512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512,512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(2048, num_classes)     
        )   

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class TemporalStreamConvNet(nn.Module):
    def __init__(self, num_classes=101):
        super(TemporalStreamConvNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(20, 96, kernel_size=7, stride=2),
            nn.LocalResponseNorm(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.LocalResponseNorm(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class TwoStreamConvNet(nn.Module):
    def __init__(self, num_classes=101):
        super(TwoStreamConvNet, self).__init__()
        self.spatial_stream = SpatialStreamConvNet(num_classes)
        self.temporal_stream = TemporalStreamConvNet(num_classes)

    def forward(self, spatial_input, temporal_input):
        spatial_output = self.spatial_stream(spatial_input)
        temporal_output = self.temporal_stream(temporal_input)
        combined_output = (spatial_output + temporal_output) / 2
        return combined_output


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
model = TwoStreamConvNet().to(device)