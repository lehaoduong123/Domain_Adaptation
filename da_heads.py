# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from gradient_reversal import GradientScalarLayer


def consistency_loss(img_feas, ins_fea, ins_labels, size_average=True):
    """
    Consistency regularization as stated in the paper
    `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
    L_cst = \sum_{i,j}||\frac{1}{|I|}\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
    """
    loss = []
    len_ins = ins_fea.size(0)
    intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
    for img_fea_per_level in img_feas:
        N, A, H, W = img_fea_per_level.shape
        img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
        img_feas_per_level = []
        assert N==2, \
            "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
        for i in range(N):
            img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
            img_feas_per_level.append(img_fea_mean)
        img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
        loss_per_level = torch.abs(img_feas_per_level - ins_fea)
        loss.append(loss_per_level)
    loss = torch.cat(loss, dim=1)
    if size_average:
        return loss.mean()
    return loss.sum()


class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, *args):
        pass

    def __call__(self, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        masks = targets.to(torch.bool)
        
        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
        
        da_img_flattened = torch.cat(da_img_flattened, dim=1)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=1)

        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)

        return da_img_loss, da_ins_loss, da_consist_loss


def make_da_heads_loss_evaluator(*args):
    loss_evaluator = DALossComputation(*args)
    return loss_evaluator


class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self):
        super(DomainAdaptationModule, self).__init__()

        self.img_weight = ...
        self.ins_weight = ...
        self.cst_weight = ...

        grl_weight = ...
        self.grl_img = GradientScalarLayer(-1.0*grl_weight)
        self.grl_ins = GradientScalarLayer(-1.0*grl_weight)
        self.grl_img_consist = GradientScalarLayer(1.0*grl_weight)
        self.grl_ins_consist = GradientScalarLayer(1.0*grl_weight)
        
        in_channels = ...
        num_ins_inputs = ...
        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)

        self.loss_evaluator = make_da_heads_loss_evaluator()

    def forward(self, img_features, da_ins_feature, da_ins_labels, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): domain labels

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        if not self.training: return {}
        
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss
            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}
