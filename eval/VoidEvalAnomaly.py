# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import torch.nn as nn
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from addtionalModels.enet import ENet
from addtionalModels.bisenetv1 import BiSeNetV1
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score



seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--model', default = "ErfNet", help= "Choose a model between ErfNet and ENet")
    parser.add_argument('--loadModel', default="enet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--discriminant',default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--temperature', default=1)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    

    #model = ERFNet(NUM_CLASSES)
    if args.model == 'ErfNet':
        model_name = 'ErfNet'
        model = ERFNet(NUM_CLASSES)
        modelpath = args.loadDir + args.loadModel
        weightspath = "../save/trainingdataerfnet/model_best_erfnet.pth"#args.loadDir + "erfnet_pretrained.pth" #args.loadWeights

        #weightspath = args.loadDir + "erfnet_pretrained.pth" #args.loadWeights

        print ("Loading model: " + modelpath)
        print ("Loading weights: " + weightspath)
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
        
    elif args.model == 'ENet':
        model_name = 'ENet'
        model = ENet(NUM_CLASSES)
        modelpath = args.loadDir + args.loadModel
        
        weightspath = "../save/trainingdataenet/model_best_enet.pth"#args.loadDir + "erfnet_pretrained.pth" #args.loadWeights
# weightspath = args.loadDir + "enet_pretrained" #args.loadWeights

        print ("Loading model: " + modelpath)
        print ("Loading weights: " + weightspath)
        state_dict = torch.load(weightspath)
        model.load_state_dict(state_dict['state_dict'])
    elif args.model == 'BiseNet':
        model_name = 'BiseNet'
        model = BiSeNetV1(NUM_CLASSES)
        modelpath = args.loadDir + args.loadModel
        weightspath = "../save/trainingdatabisenetv1/model_best_bisenetv1.pth" #args.loadDir + "erfnet_pretrained.pth" #args.loadWeights
#weightspath = args.loadDir + "bisenetv1_cityscapes.pth" #args.loadWeights

        print ("Loading model: " + modelpath)
        print ("Loading weights: " + weightspath)
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    else:
        raise ValueError("Cannot find model")

    print ("Model and weights LOADED successfully")
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()
    model.eval()
    count = 0
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        if 'FS_LostFound_full' in path:
            count+=1
            if count > len(os.listdir(args.input[0]))*0.7:
                break
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            if args.model == 'BiseNet':
              result = model(images)[0]
            else:  
              result = model(images)
        
        if (args.discriminant == "maxlogit"):
          anomaly_result = -(np.max(result.squeeze(0).data.cpu().numpy(), axis=0))
        elif (args.discriminant == "msp"):
          softmax_probs = torch.nn.functional.softmax(result.squeeze(0) / float(args.temperature), dim=0)
          anomaly_result = 1.0 - (np.max(softmax_probs.data.cpu().numpy(), axis=0))
        elif (args.discriminant == "maxentropy"):
          max_entropy = (-torch.sum(torch.nn.functional.softmax(result.squeeze(0), dim=0) * torch.nn.functional.log_softmax(result.squeeze(0), dim=0), dim=0))
          max_entropy = torch.div(max_entropy, torch.log(torch.tensor(result.shape[1])))
          anomaly_result = max_entropy.data.cpu().numpy()  
        else: 
          anomaly_result = result.squeeze(0)    
                 
        anomaly_result = result.squeeze(0).data.cpu().numpy()[19,:,:] # background as anomaly
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

    file.write( "\n")

    path = args.input[0]
    dataset = ""
    if "RoadAnomaly21" in path:
        dataset = "RoadAnomaly21"
    elif "RoadObsticle21" in path:
        dataset = "RoadObsticle21"
    elif "FS_LostFound_full" in path:
        dataset = "FS_LostFound_full"
    elif "fs_static" in path:
        dataset = "fs_static"
    elif "RoadAnomaly" in path:
        dataset = "RoadAnomaly"

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

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"{dataset}-{model_name}")
    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')
    file.write((f"                 {dataset}-{model_name}-temperature:{args.temperature}        "))
    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()