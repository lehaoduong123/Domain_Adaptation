import torchvision
import torch
import torch.nn as nn
from da_heads import DomainAdaptationModule


class DA_model(nn.Module):
    def __init__(self, n_classes, load_source_model=False) -> None:
        super().__init__()

        # setup main object detection model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(...)
        # - you can modify the backbone on your own
        # self.model.backbone = ...
        # - modify n_classes by replacing the box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes) 
        
        if load_source_model:
            msg = self.model.load_state_dict(...)
            print(msg)

        self.create_hooks()
        self.da_model = DomainAdaptationModule()
    
    def create_hooks(self):
        # setup hooks for hooking features needed by domain adaptation
        self.backbone_features = []
        self.box_features = []

        self.hook1 = self.model.backbone.register_forward_hook(lambda module, input, output: self.backbone_features.append(output))
        self.hook2 = self.model.roi_heads.box_head.register_forward_hook(lambda module, input, output: self.box_features.append(output))

    def gen_instance_level_domain_labels(self, domain_labels):
        # TODO
        ...
        
    def forward(self, x1, gt1=None, x2=None):
        self.backbone_features = []
        self.box_features = []
        
        if self.training:
            losses = {}
            # TODO: forward x1(source data) and x2(target data) then compute OD losses of x1
            od_losses = 

            
            # TODO: complete DA losses
            domain_labels = torch.cat([torch.zeros(1), torch.ones(1)]).long().to(x1[0].device)
            instance_level_domain_labels = self.gen_instance_level_domain_labels(domain_labels)
            backbone_feature_list = []
            # After the forwarding of x1 and x2
            # self.backbone_features will be [features of x1, features of x2] due to the hook function
            for i, (f1, f2) in enumerate(zip(self.backbone_features[0].values(), self.backbone_features[1].values())):
                # just use P-5 features
                if i != 3: continue
                backbone_feature_list.append(torch.cat([f1, f2], dim=0)) # [2, C, H, W]
                
            da_losses = self.da_model(img_features=backbone_feature_list, 
                                        da_ins_feature=...,
                                        da_ins_labels=instance_level_domain_labels,
                                        targets=domain_labels)
            
            losses.update(od_losses)
            losses.update(da_losses)
            
            return losses
        else:
            outputs = self.model(x1)
            return outputs # detections


