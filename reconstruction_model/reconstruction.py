#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import torch.nn.modules.transformer as transformer

from reconstruction_model.common import subtract_pose, unflatten_two, flatten_two

from habitat.core.logging import logger


def process_image(img):
    """Apply imagenet normalization to a batch of images.
    """
    # img - (bs, C, H, W)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_proc = img.float() / 255.0

    img_proc[:, 0] = (img_proc[:, 0] - mean[0]) / std[0]
    img_proc[:, 1] = (img_proc[:, 1] - mean[1]) / std[1]
    img_proc[:, 2] = (img_proc[:, 2] - mean[2]) / std[2]

    return img_proc

class View(nn.Module):
    def __init__(self, *shape):
        # shape is a list
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class FeatureReconstructionModule(nn.Module):
    """An encoder-decoder model based on transformers for reconstructing
    concepts at a target location.
    """

    def __init__(self, nfeats, noutputs, nlayers=4):
        super().__init__()
        encoder_layer = transformer.TransformerEncoderLayer(nfeats + 16, 2, nfeats)
        decoder_layer = transformer.TransformerDecoderLayer(nfeats + 16, 2, nfeats)
        self.encoder = transformer.TransformerEncoder(encoder_layer, nlayers)
        self.decoder = transformer.TransformerDecoder(decoder_layer, nlayers)
        self.predict_outputs = nn.Linear(nfeats + 16, noutputs)

    def forward(self, x):
        """
        Inputs:
            x - dictionary consisting of the following:
            {
                'history_image_features': (T, N, nfeats)
                'history_pose_features': (T, N, 16)
                'target_pose_features': (1, N, 16)
            }
        Outputs:
            pred_outputs - (1, N, noutputs)
        """
        target_pose_features = x["target_pose_features"][0]
        T, N, nfeats = x["history_image_features"].shape
        nRef = target_pose_features.shape[1]
        device = x["target_pose_features"].device
        # =================== Encode features and poses =======================
        encoder_inputs = torch.cat(
            [x["history_image_features"], x["history_pose_features"]], dim=2
        )  # (T, N, nfeats+16)
        encoded_features = self.encoder(encoder_inputs)  # (T, N, nfeats+16)
        # ================ Decode features for given poses ====================
        decoder_pose_features = target_pose_features.unsqueeze(0)  # (1, N, 16)
        # Initialize as zeros
        decoder_image_features = torch.zeros(
            *decoder_pose_features.shape[:2], nfeats
        ).to(
            device
        )  # (1, N, nfeats)
        decoder_inputs = torch.cat(
            [decoder_image_features, decoder_pose_features], dim=2
        )  # (1, N, nfeats+16)
        decoder_features = self.decoder(
            decoder_inputs, encoded_features
        )  # (1, N, nfeats+16)
        pred_outputs = self.predict_outputs(decoder_features).squeeze(0)
        return pred_outputs.unsqueeze(0)


class FeatureNetwork(nn.Module):
    """Network to extract image features.
    """

    def __init__(self):
        super().__init__()
        resnet = tmodels.resnet50(pretrained=True)
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

    def forward(self, x):
        #logger.info(f"x={x.shape}")
        feat = self.net(x).squeeze(3).squeeze(2)
        feat = F.normalize(feat, p=2, dim=1)
        return feat


class PoseEncoder(nn.Module):
    """Network to encode pose information.
    """

    def __init__(self):
        super().__init__()
        #logger.info("Init PoseEncoder")
        self.main = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16),)

    def forward(self, x):
        #logger.info("Before PoseEncoder")
        return self.main(x)

def multi_label_classification_loss(x, y, reduction="batchmean"):
    """
    Multi-label classification loss - KL divergence between a uniform
    distribution over the GT classes and the predicted probabilities.
    Inputs:
        x - (bs, nclasses) predicted logits
        y - (bs, nclasses) with ones for the right classes and zeros
            for the wrong classes
    """
    x_logprob = F.log_softmax(x, dim=1)
    y_prob = F.normalize(
        y.float(), p=1, dim=1
    )  # L1 normalization to convert to probabilities
    loss = F.kl_div(x_logprob, y_prob, reduction=reduction)
    return loss


def rec_loss_fn_classify(
    x_logits, x_gt_feat, cluster_centroids, K=5, reduction="batchmean"
):
    """
    Given the predicted logits and ground-truth reference feature,
    find the top-K NN cluster centroids to the ground-truth feature.
    Using the top-k clusters as the ground-truth, use a multi-label
    classification loss.

    NOTE - this assumes that x_gt_feat and cluster_centroids are unit vectors.

    Inputs:
        x_logits - (bs, nclusters) predicted logits
        x_gt_feat - (bs, nclusters) reference feature that consists of
                    similarity scores between GT image and cluster centroids.
        cluster_centroids - (nclusters, feat_size) cluster centroids
    """
    bs, nclasses = x_logits.shape
    nclusters, feat_size = cluster_centroids.shape
    device = x_logits.device

    # Compute cosine similarity between x_gt_feat and cluster_centroids
    cosine_sim = x_gt_feat

    # Sample top-K similar clusters
    topK_outputs = torch.topk(cosine_sim, K, dim=1)

    # Generate K-hot encoding
    k_hot_encoding = (
        torch.zeros(bs, nclasses).to(device).scatter_(1, topK_outputs.indices, 1.0)
    )

    loss = multi_label_classification_loss(
        x_logits, k_hot_encoding, reduction=reduction
    )

    return loss


def compute_reconstruction_rewards(
    obs_feats,
    obs_odometer,
    tgt_feats,
    tgt_poses,
    cluster_centroids_t,
    decoder,
    pose_encoder,
):
    """
    Inputs:
        obs_feats           - (T, N, nclusters)
        obs_odometer        - (T, N, 3) --- (y, x, theta)
        tgt_feats           - (N, nRef, nclusters)
        tgt_poses           - (N, nRef, 3) --- (y, x, theta)
        cluster_centroids_t - (nclusters, feat_dim)
        decoder             - decoder model
        pose_encoder        - pose_encoder model

    Outputs:
        reward              - (N, nRef) float values indicating how many
                              GT clusters were successfully retrieved for
                              each target.
    """
    #logger.info("Start compute_reconstruction_rewards")
    T, N, nclusters = obs_feats.shape
    nRef = tgt_feats.shape[1]
    device = obs_feats.device

    #logger.info("before obs_feats_exp")
    obs_feats_exp = obs_feats.unsqueeze(2)
    obs_feats_exp = obs_feats_exp.expand(
        -1, -1, nRef, -1
    ).contiguous()  # (T, N, nRef, nclusters)
    #logger.info("obs_feats_exp 1")
    obs_odometer_exp = obs_odometer.unsqueeze(2)
    #logger.info("obs_feats_exp 2")
    obs_odometer_exp = obs_odometer_exp.expand(
        -1, -1, nRef, -1
    ).contiguous()  # (T, N, nRef, 3)
    #logger.info("obs_feats_exp 3")
    tgt_poses_exp = (
        tgt_poses.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
    )  # (T, N, nRef, 3)
    #logger.info("obs_feats_exp 4")

    # Compute relative poses
    #logger.info("before obs_odometer_exp")
    obs_odometer_exp = obs_odometer_exp.view(T * N * nRef, 3)
    #logger.info("obs_odometer_exp 1")
    tgt_poses_exp = tgt_poses_exp.view(T * N * nRef, 3)
    #logger.info("obs_odometer_exp 2")
    obs_relpose = subtract_pose(
        obs_odometer_exp, tgt_poses_exp
    )  # (T*N*nRef, 3) --- (x, y, phi)
    #logger.info("obs_odometer_exp 3")
    #logger.info(f"obs_relpose_device={obs_relpose.device}")

    # Compute pose encoding
    #logger.info("before obs_relpose_enc")
    with torch.no_grad():
        #logger.info("before obs_relpose_enc 111")
        #logger.info(f"pose_encoder={pose_encoder}")
        #logger.info(f"pose_encoder_device={pose_encoder.device}")
        #logger.info(f"obs_relpose shape: {obs_relpose.shape}")
        obs_relpose_enc = pose_encoder(obs_relpose)  # (T*N*nRef, 16)
    #logger.info("obs_relpose_enc 1")
    obs_relpose_enc = obs_relpose_enc.view(T, N, nRef, -1)  # (T, N, nRef, 16)
    #logger.info("obs_relpose_enc 2")
    tgt_relpose_enc = torch.zeros(1, *obs_relpose_enc.shape[1:]).to(
        device
    )  # (1, N, nRef, 16)
    #logger.info("obs_relpose_enc 3")
    
    # Compute reconstructions
    #logger.info("obs_feats_exp 1")
    obs_feats_exp = obs_feats_exp.view(T, N * nRef, nclusters)
    #logger.info("obs_feats_exp 2")
    obs_relpose_enc = obs_relpose_enc.view(T, N * nRef, -1)
    #logger.info("obs_feats_exp 3")
    tgt_relpose_enc = tgt_relpose_enc.view(1, N * nRef, -1)

    #logger.info("rec_inputs 1")
    rec_inputs = {
        "history_image_features": obs_feats_exp,
        "history_pose_features": obs_relpose_enc,
        "target_pose_features": tgt_relpose_enc,
    }
    #logger.info("rec_inputs 2")

    with torch.no_grad():
        pred_logits = decoder(rec_inputs)  # (1, N*nRef, nclusters)
    #logger.info("pred_logits 1")
    pred_logits = pred_logits.squeeze(0)  # (N*nRef, nclusters)
    #logger.info("pred_logits 2")
    pred_logits = unflatten_two(pred_logits, N, nRef)  # (N, nRef, nclusters)
    #logger.info("pred_logits 3")
    
    # Compute GT classes
    #logger.info("tgt_feats_sim 1")
    tgt_feats_sim = tgt_feats  # (N, nRef, nclusters)
    #logger.info("tgt_feats_sim 2")
    topk_gt = torch.topk(tgt_feats_sim, 5, dim=2)
    #logger.info("tgt_feats_sim 3")
    topk_gt_values = topk_gt.values  # (N, nRef, nclusters)
    #logger.info("tgt_feats_sim 4")
    topk_gt_thresh = topk_gt_values.min(dim=2).values  # (N, nRef)
    #logger.info("tgt_feats_sim 5")

    # ------------------ KL Div loss based reward --------------------
    #logger.info("KL Div loss 1")
    reward = -rec_loss_fn_classify(
        flatten_two(pred_logits),
        flatten_two(tgt_feats),
        cluster_centroids_t.t(),
        K=2,
        reduction="none",
    ).sum(
        dim=1
    )  # (N*nRef, )
    #logger.info("KL Div loss 2")
    reward = reward.view(N, nRef)
    #logger.info("KL Div loss 3")

    return reward


def masked_mean(values, masks, axis=None):
    return (values * masks).sum(axis=axis) / (masks.sum(axis=axis) + 1e-10)