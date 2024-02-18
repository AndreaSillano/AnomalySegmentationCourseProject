import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

NUM_CLASSES = 20


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
class LogitNormLoss(nn.Module):

    def __init__(self, weight,t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.weight = weight
    def forward(self, output, target):
        norms =  output/(torch.norm(output, p=2, dim=1, keepdim=True) + 1e-7)
        output_prob =F.softmax(norms / self.t, dim=1)


        return F.cross_entropy(output_prob, target, weight = self.weight)
    
class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes =NUM_CLASSES, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        features =torch.transpose(features, 1,3 )
        distances = torch.abs(self.distance_scale.cuda()) * torch.cdist(F.normalize(features).cuda(), F.normalize(self.prototypes).cuda(), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        distances =torch.transpose(distances, 3,1)
        logits = -distances
        return logits / self.temperature



class IsoMaxPlusLoss(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLoss, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################

        

        '''distances = -logits
        targets = targets.view(distances.size(0), -1)
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()'''
        
        step1 = IsoMaxPlusLossFirstPart(20)
        logits = step1.forward(logits)

        distances = -logits
        targets = targets.view(distances.size(0), -1)

        # Softmax operation
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)

        # Flatten probabilities_for_training for indexing
        probabilities_flat = probabilities_for_training.view(distances.size(0), -1)

        # Gather probabilities at target indices
        probabilities_at_targets = probabilities_flat.gather(1, targets)

        # Compute loss
        loss = -torch.log(probabilities_at_targets).mean()
        return loss






