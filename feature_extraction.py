import torch
import torch.nn as nn



class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, x):
        """
        Expect x to be an image of size (bs, 3, height, width)
        Output should be the feature vector of image x
        with size (bs, f, height, width)
        """
        feature_vector = None
        cf = self.color_feature(x)
        pf= self.position_feature(x).unsqueeze(0)
        feature_vector = torch.concat([cf, pf], dim=1)

        return feature_vector

    def color_feature(self, x):
        return x
    def position_feature(self, img):
        _, _, height, width = img.shape
        xs = torch.arange(0,width)
        ys = torch.arange(0,height)
        return torch.stack(torch.meshgrid([ys, xs], indexing="ij"))
    