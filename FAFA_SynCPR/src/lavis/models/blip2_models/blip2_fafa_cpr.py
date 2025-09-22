"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures


@registry.register_model("blip2_fafa_cpr")
class Blip2FAFACPR(Blip2Base):
    """
    FAFA (Fine-grained Adaptive Feature Alignment) model for Composed Person Retrieval.
    Based on BLIP-2 architecture with modifications for CPR task.
    """
    """
    BLIP2 first-stage model with Q-former and ViT.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=128,
        use_hnnce=False,
        use_mlm=True,
        use_mmd=False,
        use_coral=False,
        use_sdm=True,
        use_dsu=True,
        use_soft=True,
        use_dispersion = True,
        fda_k=6,  # Number of top-k features for FDA
        fda_alpha=0.5,  # Soft label strength in FDA
        fd_margin=0.5  # Margin parameter for Feature Diversity
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.use_dispersion = use_dispersion  # Feature Diversity (FD) loss
        self.dispersion_loss = FeatureDispersionLoss(margin=fd_margin)
        self.fda_k = fda_k  # Number of top-k features for FDA
        self.fda_alpha = fda_alpha  # Soft label strength

        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Freeze vision encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        # Additional technique
        self.use_hnnce = use_hnnce
        if use_hnnce:
            self.hnnce_loss = HardNegativeNCE()
        
        self.use_mlm = use_mlm
        if use_mlm:
            self.mlm_predictor= MaskedFeaturePrediction(feature_dim=self.Qformer.config.hidden_size)

        self.use_coral = use_coral
        self.use_mmd = use_mmd
        self.use_sdm = use_sdm
        self.use_dsu = use_dsu
        self.use_soft = use_soft
        if use_dsu:
            self.dg = DistributionUncertainty()
    
    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        if self.use_sdm:
            qid = samples["query_id"]
            # print(qid.shape, qid)
        loss_ret = dict()

        ###============== Reference Text Fusion ===================###
        # reference image feature  
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        # fusion reference image and text tokens into a set of multi-modal tokens
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # print(fusion_output.shape)
        fusion_feats = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]), dim=-1)
        # print(fusion_feats.shape)
        ###============== Fusion-target Contrastive ===================###
        # reference image feature  
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(image.device)
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        
        target_feats = F.normalize(self.vision_proj(self.dg(target_output.last_hidden_state) if self.use_dsu else target_output.last_hidden_state), dim=-1)
        # print(fusion_feats.shape, target_feats.shape)
        # in-batch ce -> info-nce
        # print(fusion_feats.unsqueeze(1).unsqueeze(1).shape, target_feats.permute(0, 2, 1).shape)
        if self.use_dispersion:
            # print(target_feats.shape)
            loss_fd = self.dispersion_loss(target_feats)  # Feature Diversity loss
            loss_ret.update({'loss_fd': loss_fd})
            
        sim_t2q = torch.matmul(fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)).squeeze()

        if self.use_soft:
            sim_i2t, _ = torch.topk(sim_t2q, k=self.fda_k, dim=-1)  # FDA with k=6 from paper
            sim_i2t = sim_i2t.mean(-1)
        else:
            sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp

        # FDA (Fine-grained Dynamic Alignment) loss
        if self.use_hnnce:
            loss_fda = self.hnnce_loss(sim_i2t)
        elif self.use_sdm:
            loss_fda = compute_sdm(sim_i2t, qid, factor=1)
        else:
            bs = image.size(0)
            targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(image.device)
            loss_fda = F.cross_entropy(sim_i2t, targets)
        loss_ret.update({'loss_fda': loss_fda})

        if self.use_mlm:
            # MFR (Masked Feature Reasoning) loss
            loss_mfr = self.mlm_predictor(fusion_output.last_hidden_state[:, self.num_query_token, :], \
                    target_output.last_hidden_state.mean(dim=1))
            loss_ret.update({'loss_mfr': loss_mfr})
        
        if self.use_mmd:
            loss_mmd = mmd_loss(fusion_output.last_hidden_state[:, self.num_query_token, :], \
                    target_output.last_hidden_state.mean(dim=1))
            loss_ret.update({'loss_mmd': loss_mmd})

        if self.use_coral:
            loss_coral = coral_loss(fusion_output.last_hidden_state[:, self.num_query_token, :], \
                    target_output.last_hidden_state.mean(dim=1))
            loss_ret.update({'loss_coral': loss_coral})
        
        return loss_ret

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def forward_query(self, image, text):
        # reference image feature  
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        # fusion reference image and text tokens into a set of multi-modal tokens
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        query_embeddings = fusion_output.last_hidden_state[:, self.num_query_token, :]
        return F.normalize(self.text_proj(query_embeddings), dim=-1)
    
    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, :self.num_query_token, :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit
    
    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]), dim=-1
        )
        

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        # Use the same FDA k as in training for consistency
        if self.use_soft:
            sim_i2t, _ = torch.topk(sim_t2q, k=self.fda_k, dim=-1)  # FDA with k from config
            sim_i2t = sim_i2t.mean(-1)
        else:
            sim_i2t, _ = sim_t2q.max(-1)
        return sim_i2t

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_embeddings = output.last_hidden_state[:, self.num_query_token, :]
            multimodal_embeds = F.normalize(self.text_proj(query_embeddings), dim=-1)
            # print(multimodal_embeds.shape)

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)


class HardNegativeNCE(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, **kwargs):
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, sim_matrix):
        batch_size = sim_matrix.size(0)
        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = ((batch_size - 1) * torch.exp(beta_sim) / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim))))
        
        w_t2v = ((batch_size - 1) * torch.exp(beta_sim) / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim))))
        
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (denominator_t2v - nominator).mean()
        return hn_nce_loss


class MaskedFeaturePrediction(nn.Module):
    def __init__(self, feature_dim, mask_ratio=0.3, hidden_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio

        self.predictor = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def mask_features(self, features):
        B, D = features.shape
        mask = (torch.rand(B, D, device=features.device) < self.mask_ratio).float()
        masked_features = features * (1 - mask)
        return masked_features, mask

    def forward(self, feat_A, feat_B):
        masked_A, mask_A = self.mask_features(feat_A)
        pred_A = self.predictor(torch.cat([masked_A, feat_B], dim=1) if feat_B is not None else masked_A)

        masked_B, mask_B = self.mask_features(feat_B)
        pred_B = self.predictor(torch.cat([masked_B, feat_A], dim=1) if feat_A is not None else masked_B)

        loss_A = F.mse_loss(pred_A * mask_A, feat_A * mask_A, reduction='sum') / mask_A.sum()
        loss_B = F.mse_loss(pred_B * mask_B, feat_B * mask_B, reduction='sum') / mask_B.sum()

        return (loss_A + loss_B) / 2


def mmd_loss(x, y, gamma=1.0):
    def rbf_kernel(x, y):
        x_sqnorms = torch.sum(x**2, dim=1, keepdim=True)
        y_sqnorms = torch.sum(y**2, dim=1, keepdim=True)
        pairwise_dist = x_sqnorms - 2*torch.matmul(x, y.T) + y_sqnorms.T
        return torch.exp(-gamma * pairwise_dist)
    
    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    return (k_xx.mean() + k_yy.mean() - 2*k_xy.mean())


def coral_loss(x, y):
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_y = (y.T @ y) / (y.size(0) - 1)
    diff = cov_x - cov_y
    return torch.norm(diff, p='fro')**2 / (4 * x.size(1)**2)

def compute_sdm(sim_matrix, qid, epsilon=1e-8, factor=1):
    """
    Similarity Distribution Matching
    """
    batch_size = sim_matrix.shape[0]
    qid = qid.reshape((batch_size, 1))
    qid_dist = qid - qid.t()
    labels = (qid_dist == 0).float() * factor
    labels = labels * (1 - torch.eye(batch_size)) + torch.eye(batch_size)
    labels = labels.to(sim_matrix.device)

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(sim_matrix, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(sim_matrix, dim=1) - torch.log(labels_distribute + epsilon))

    t2i_pred = F.softmax(sim_matrix.t(), dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(sim_matrix.t(), dim=1) - torch.log(labels_distribute + epsilon))

    loss = (torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))) / 2

    return loss

class FeatureDispersionLoss(nn.Module):
    def __init__(self, margin=0.5, sample_size=1024):
        super(FeatureDispersionLoss, self).__init__()
        self.margin = margin
        self.sample_size = sample_size  # 限制最多采样多少点来计算 dispersion

    def forward(self, features):
        """
        features: Tensor of shape [batch_size, num_query_tokens, feature_dim]
        """
        B, T, D = features.shape
        features = features.view(B * T, D)  # [batch_size * num_tokens, feature_dim]
        
        if features.size(0) > self.sample_size:
            indices = torch.randperm(features.size(0), device=features.device)[:self.sample_size]
            features = features[indices]

        # compute pairwise cosine similarity
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_matrix.size(0), device=features.device).bool()
        sim_matrix.masked_fill_(mask, -1.0)

        loss = F.relu(sim_matrix - self.margin).mean()
        return loss

class DistributionUncertainty(nn.Module):
    def __init__(self, p=0.8, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x, dim=0):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        _, l, d = x.shape

        mean = x.mean(dim=dim, keepdim=False)
        std = (x.var(dim=dim, keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(1, l, d)) / std.reshape(1, l, d)
        x = x * gamma.reshape(1, l, d) + beta.reshape(1, l, d)

        return x