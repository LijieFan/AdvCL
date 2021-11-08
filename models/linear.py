from torch import nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', feat_dim=512, num_classes=10):
        super(LinearClassifier, self).__init__()
        # _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class NonLinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', feat_dim=512, num_classes=10):
        super(NonLinearClassifier, self).__init__()
        # _, feat_dim = model_dict[name]
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        # self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        # self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        # features = F.relu(self.bn1(self.fc1(features)))
        # features = F.relu(self.bn2(self.fc2(features)))
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc3(features)