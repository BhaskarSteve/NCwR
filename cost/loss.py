import torch.nn as nn

class CwRLoss(nn.Module):
    def __init__(self, c, num_classes):
        super(CwRLoss, self).__init__()
        self.c = c
        self.num_classes = num_classes

    def forward(self, output, target):
        label_rej = 0 * target + self.num_classes
        loss_fn = nn.CrossEntropyLoss()
        loss1 = loss_fn(output, target)
        loss2 = loss_fn(output, label_rej)
        loss = (loss1 + ((1 - self.c) * loss2))
        return loss