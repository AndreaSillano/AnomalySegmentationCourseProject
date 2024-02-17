import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 20
class LogitNormLoss(nn.Module):

    def __init__(self, weight,t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.weight = weight
    def forward(self, output, target):
        norms =  output/(torch.norm(output, p=2, dim=1, keepdim=True) + 1e-7)
        output_prob =F.softmax(norms / self.t, dim=1)
        print(output_prob.shape)
        print(target.shape)

        return F.cross_entropy(output_prob, target, weight = self.weight)
    
class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
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
        last_layer = IsoMaxPlusLossFirstPart(logits.size(1), NUM_CLASSES)
        logits = last_layer.forward(logits)
        
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        return loss