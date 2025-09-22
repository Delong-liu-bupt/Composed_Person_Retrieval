from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from prettytable import PrettyTable
from utils import collate_fn, device

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def batchwise_similarity(qfeats, gfeats, batch_size=500):
    qfeats = qfeats.unsqueeze(1).unsqueeze(1)  # [2202, 1, 1, 256]
    gfeats = gfeats.permute(0, 2, 1)           # [20510, 256, 32]

    num_q = qfeats.size(0)
    num_g = gfeats.size(0)

    sim_t2q = torch.empty((num_q, num_g, 32), device=qfeats.device)

    for q_start in range(0, num_q, batch_size):
        q_end = min(q_start + batch_size, num_q)
        q_batch = qfeats[q_start:q_end]  # [batch_size, 1, 1, 256]

        for g_start in range(0, num_g, batch_size):
            g_end = min(g_start + batch_size, num_g)
            g_batch = gfeats[g_start:g_end]  # [batch_size, 256, 32]

            # (q_batch: [bq, 1, 1, 256]) x (g_batch: [bg, 256, 32])
            # -> (bq, bg, 1, 32) -> (bq, bg, 32)
            sim_batch = torch.matmul(q_batch, g_batch).squeeze(2)

            # 保存计算结果
            sim_t2q[q_start:q_end, g_start:g_end] = sim_batch

    return sim_t2q  # [2202, 20510, 32]

def compute_ticpr_val_metrics(blip_model, val_query_set, val_gallery_set, txt_processors, soft=False):
    device = next(blip_model.parameters()).device
    # blip_model = blip_model.half()
    gallery_loader = DataLoader(dataset=val_gallery_set, batch_size=64, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    query_loader = DataLoader(dataset=val_query_set, batch_size=64, num_workers=2,
                                    pin_memory=True, collate_fn=collate_fn)
    qids, gids, qfeats, gfeats = [], [], [], []
    print("Computing image gallery features.")
    for iid, img in tqdm(gallery_loader):
        img = img.to(device)
        with torch.no_grad():
            image_features, _ = blip_model.extract_target_features(img.half(), mode="mean")
        gids.append(iid.view(-1)) # flatten 
        gfeats.append(image_features)
    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)
    # print(gfeats.shape)
    for iid, img, captions in tqdm(query_loader):
        img = img.to(device)
        # captions = captions.to(device)
        captions: list = np.array(captions).T.flatten().tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.no_grad():
            query_features = blip_model.extract_features({"image": img.half(), "text_input": captions})
            query_features =query_features.multimodal_embeds

        qids.append(iid.view(-1)) # flatten 
        qfeats.append(query_features)
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)
    # print(qfeats.unsqueeze(1).unsqueeze(1).shape, gfeats.permute(0, 2, 1).shape)
    # gfeats = F.normalize(gfeats, p=2, dim=1) # image features
    # qfeats = F.normalize(qfeats, p=2, dim=1)
    
    sim_t2q = batchwise_similarity(qfeats, gfeats, batch_size=500)
    if soft:
        # Use the same k value as configured in the model
        fda_k = getattr(blip_model, 'fda_k', 6)  # Default to 6 if not set
        similarity, _ = torch.topk(sim_t2q, k=fda_k, dim=-1)
        similarity = similarity.mean(-1)
    else:
        similarity, _ = sim_t2q.max(-1)

    com_cmc, com_mAP, com_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    com_cmc, com_mAP, com_mINP = com_cmc.numpy(), com_mAP.numpy(), com_mINP.numpy()
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
    table.add_row(['com', com_cmc[0], com_cmc[4], com_cmc[9], com_mAP, com_mINP])

    # table.float_format = '.4'
    table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
    print('\n' + str(table))

    return com_cmc[0], com_cmc[4], com_cmc[9], com_mAP

