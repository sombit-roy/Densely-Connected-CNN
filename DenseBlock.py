import torch.nn as nn

class DenseBlock(nn.Module):
    
    def __init__(self, nb_layers, in_planes, growth_rate, block):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers)
        
    def _make_layer(self, block, in_planes, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)