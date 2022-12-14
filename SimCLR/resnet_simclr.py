import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):

    def __init__(self, out_dim):
        super(ResNetSimCLR, self).__init__()

        self.backbone = models.resnet50(pretrained=False, num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features

        # -> linear head to 2 layer non-linear mlp head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

if __name__ == '__main__':
    model = ResNetSimCLR(out_dim=128)
    import pdb
    pdb.set_trace()