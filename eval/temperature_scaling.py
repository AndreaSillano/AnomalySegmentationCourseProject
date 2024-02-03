import torch
from torch import is_same_size, nn, optim
from torch.nn import functional as F
import numpy as np 
from PIL import Image
from torch.autograd import Variable
import glob
import os
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, images_path):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()
        ood_gts_list =[]
        anomaly_score_list =[]
        for path in glob.glob(os.path.expanduser(images_path)):
          print(path)
          images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
          images = images.permute(0,3,1,2).cuda()
          with torch.no_grad():
              result = self.model(images)

          #anomaly_result = result
          softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
          anomaly_result = 1.0 - (np.max(softmax_probs.data.cpu().numpy(), axis=0))
                
          pathGT = path.replace("images", "labels_masks")                
          if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
          if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")                
          if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")  

          mask = Image.open(pathGT)
          ood_gts = np.array(mask)

          if "RoadAnomaly" in pathGT:
              ood_gts = np.where((ood_gts==2), 1, ood_gts)
          if "LostAndFound" in pathGT:
              ood_gts = np.where((ood_gts==0), 255, ood_gts)
              ood_gts = np.where((ood_gts==1), 0, ood_gts)
              ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

          if "Streethazard" in pathGT:
              ood_gts = np.where((ood_gts==14), 255, ood_gts)
              ood_gts = np.where((ood_gts<20), 0, ood_gts)
              ood_gts = np.where((ood_gts==255), 1, ood_gts)

          if 1 not in np.unique(ood_gts):
              continue              
          else:
              ood_gts_list.append(ood_gts)
              anomaly_score_list.append(anomaly_result)

          del result, anomaly_result, ood_gts, mask
          torch.cuda.empty_cache()


        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))
        
        val_out = torch.from_numpy(val_out)
        val_label = torch.from_numpy(val_label)
        val_out = torch.transpose(val_out.unsqueeze(0), 0, 1).squeeze(0)
        val_label = torch.transpose(val_label.unsqueeze(0), 0, 1).squeeze(0)
        print(val_out)
        #logits = torch.cat(anomaly_score_list).cuda()
        #labels = torch.cat(val_label).cuda()
        
        logits = val_out
        labels = val_label
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece