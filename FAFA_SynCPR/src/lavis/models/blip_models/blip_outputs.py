"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BlipOutput:
    # intermediate outputs (to be used for computing losses)
    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_pooled: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_pooled: Optional[torch.FloatTensor] = None

    # outputs to be returned
    intermediate_output: Optional[torch.FloatTensor] = None

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None


@dataclass
class BlipOutputFeatures:
    """
    Data class for features from BLIP models.

    image_embeds: shape (batch_size, num_patches, embed_dim)
    image_embeds_pooled: shape (batch_size, embed_dim)
    text_embeds: shape (batch_size, seq_len, embed_dim)
    text_embeds_pooled: shape (batch_size, embed_dim)

    """

    # intermediate outputs (to be used for computing losses)
    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

    multimodal_embeds: Optional[torch.FloatTensor] = None


@dataclass
class BlipSimilarity:
    """
    Data class for similarity from BLIP models.

    sim_i2t: shape (batch_size_image, batch_size_text)
    sim_t2i: shape (batch_size_text, batch_size_image)

    """

    sim_i2t: torch.FloatTensor
    sim_t2i: torch.FloatTensor

    sim_i2t_m: Optional[torch.FloatTensor] = None
    sim_t2i_m: Optional[torch.FloatTensor] = None

    sim_i2t_targets: Optional[torch.FloatTensor] = None
    sim_t2i_targets: Optional[torch.FloatTensor] = None


@dataclass
class BlipIntermediateOutput:
    """
    Data class for intermediate outputs from BLIP models.

    image_embeds: shape (batch_size, num_patches, embed_dim)
    text_embeds: shape (batch_size, seq_len, embed_dim)

    image_embeds_m: shape (batch_size, num_patches, embed_dim)
    text_embeds_m: shape (batch_size, seq_len, embed_dim)

    encoder_output: shape (batch_size, seq_len, embed_dim)
    encoder_output_neg: shape (batch_size, seq_len, embed_dim)

    decoder_output: shape (batch_size, seq_len, embed_dim)
    decoder_labels: shape (batch_size, seq_len)

    itm_logits: shape (batch_size * 3, 2)
    itm_labels: shape (batch_size * 3,)

    """

    # uni-modal features
    image_embeds: torch.FloatTensor
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[torch.FloatTensor] = None
    encoder_output_neg: Optional[torch.FloatTensor] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[torch.FloatTensor] = None
    decoder_labels: Optional[torch.FloatTensor] = None