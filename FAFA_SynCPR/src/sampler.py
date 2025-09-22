from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
from typing import List, Iterator

class MultiCprBatchSampler(Sampler[List[int]]):
    """
    BatchSampler that for each batch:
      - Groups samples by cpr_id
      - Randomly selects a cpr_id each time, randomly samples 1~N samples from that group (takes all if insufficient)
      - Adds these samples to the current batch until batch_size is filled or all samples are drawn
    """

    def __init__(self, dataset, batch_size: int, max_per_cpr: int, drop_last: bool = False):
        """
        Args:
            dataset: Your SynCPRDataset instance, each item in dataset.annotations needs to contain 'cpr_id'
            batch_size: Maximum number of samples per batch
            max_per_cpr: N, maximum number of samples to draw from a single cpr_id in one batch
            drop_last: Whether to drop the last batch if it's smaller than batch_size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_per_cpr = max_per_cpr
        self.drop_last = drop_last

        # Build mapping from cpr_id -> list[idx]
        self.cpr_to_indices = defaultdict(list)
        for idx, ann in enumerate(self.dataset.annotations):
            self.cpr_to_indices[ann['cpr_id']].append(idx)

        # Store all cpr_id list, will be deep copied during iteration
        self.all_cpr_ids = list(self.cpr_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        # Deep copy and shuffle each list
        cpr_to_inds = {cpid: inds.copy() for cpid, inds in self.cpr_to_indices.items()}
        for inds in cpr_to_inds.values():
            random.shuffle(inds)
        available_cpr = list(cpr_to_inds.keys())

        batch: List[int] = []
        while available_cpr:
            # If batch is full, yield it first
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

            # Randomly pick a cpr_id
            cpid = random.choice(available_cpr)
            inds = cpr_to_inds[cpid]

            # Decide how many samples to draw from this cpr_id
            k = random.randint(1, self.max_per_cpr)
            take = min(k, len(inds))

            # Extract samples
            to_add = inds[:take]
            batch.extend(to_add)
            # Update remaining samples
            cpr_to_inds[cpid] = inds[take:]
            if not cpr_to_inds[cpid]:
                # Remove this cpr_id after all samples are taken
                available_cpr.remove(cpid)

        # Loop ends, might have remaining batch
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # This value is just an estimate since the number of samples drawn each time is random
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, _, _, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdnumSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            num_instances = random.randint(1, self.num_instances)# if random.random() < 0.5 else 1#random.randint(1, self.num_instances)
            if len(idxs) < num_instances:
                idxs = np.random.choice(idxs, size=num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)


    def __len__(self):
        return self.length