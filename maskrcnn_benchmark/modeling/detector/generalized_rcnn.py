# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads
# from ..da_headc.da_heads import build_da_heads

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        features = self.backbone(images.tensors)
        # res_feat = self.backbone.body(images.tensors)
        # features = self.backbone.fpn(res_feat)

        proposals, proposal_losses = self.rpn(images, features, targets)
        da_losses = {}
        if self.roi_heads:
            x, result, detector_losses, da_ins_feas, da_ins_labels, da_proposals = self.roi_heads(features, proposals, targets)

            # import glob
            # all_mat = glob.glob('/home/yc/workplace/code/daf-dev/tmp_*.mat')
            # if len(all_mat) < 10:
            #     import scipy.io as sio
            #     sio.savemat('/home/yc/workplace/code/daf-dev/tmp_{}.mat'.format(len(all_mat)),
            #             {'labels': da_ins_labels.detach().cpu().numpy(),
            #             'ins_feat': da_ins_feas.detach().cpu().numpy(),
            #             'da_proposals_src': da_proposals[0].bbox.detach().cpu().numpy(),
            #             'da_proposals_tgt': da_proposals[1].bbox.detach().cpu().numpy()})

            if self.da_heads:
                da_losses = self.da_heads(result, features, da_ins_feas, da_ins_labels, da_proposals, targets)
                # da_losses = self.da_heads(result, res_feat, da_ins_feas, da_ins_labels, da_proposals, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses

        return result
