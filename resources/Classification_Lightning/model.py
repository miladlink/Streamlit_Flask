import torch
import torch.nn as nn
from torchvision import models


class torchModel (nn.Module):
    """ Base class for calling imagenet pretrained models """
    def __init__ (self, model_name, num_hiddens = 128, num_classes = 2, pretrained = True):
        super (torchModel, self).__init__ ()
        
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        
        # create cnn model
        cnn = models.__dict__ [model_name](pretrained)
        
        # find in features to linear model
        if hasattr (cnn, 'classifier'):
            if 'Linear' in str (cnn.classifier)[:6]:
                in_features = cnn.classifier.in_features
            else:
                in_features = cnn.classifier [-1].in_features
                
        else:
            in_features = cnn.fc.in_features
        
        fc = nn.Sequential (
                    nn.Linear (in_features, num_hiddens),
                    nn.Dropout (p = 0.2),
                    nn.BatchNorm1d (self.num_hiddens),
                    nn.LeakyReLU (0.1, inplace =  True),
                    nn.Linear (num_hiddens, num_classes)
        )
        
        # remove fc layers from cnn and add a new fc layer
        if hasattr (cnn, 'fc'):
            cnn.fc = fc
            
        else:
            cnn.classifire = fc
            
        self.cnn = cnn
        
        
    def forward (self, x):
        return self.cnn (x)
    
    
    def save_weights (self, weight_path):
        """ saving model's weights """
        print ('=> saving checkpoint')
        state = {'state_dict': model.state_dict ()}
        torch.save (state, weight_path)
        
        
    def load_weights (self, weight_path):
        """ loading model's weights """
        print ('=> loading checkpoint')
        model.load_state_dict (torch.load (weight_path) ['state_dict'])