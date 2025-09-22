import json
import os.path as op
from typing import List

from .bases import BaseDataset
from prettytable import PrettyTable

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


class ITCPRDataset(BaseDataset):
    """
    In-the-wild Test dataset for Composed Person Retrieval (ITCPR)
    Manually annotated test set for evaluating CPR models

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    """
    dataset_dir = 'Com_ReID'

    def __init__(self, root='/mnt/cache/liudelong/data/datasets', verbose=True):
        super(ITCPRDataset, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = self.dataset_dir

        self.query_path = op.join(self.dataset_dir, 'query.json')
        self.gallery_path = op.join(self.dataset_dir, 'gallery.json')
        self._check_before_run()

        self.query_annos, self.gallery_annos = self._split_anno(self.query_path, self.gallery_path)

        

        self.query, self.query_pid_container, self.query_iid_container = self._process_query(self.query_annos)
        self.gallery, self.gallery_pid_container, self.gallery_iid_container = self._process_gallery(self.gallery_annos)

        if verbose:
            self.logger.info("=> ITCPR Images and Captions are loaded")
            self.show_itcpr_info()
    
    def show_itcpr_info(self):
        num_query_pids, num_query_iids, num_query_imgs = len(
            self.query_pid_container), len(self.query_iid_container), len(self.query['img_paths'])
        num_gallery_pids, num_gallery_iids, num_gallery_imgs = len(
            self.gallery_pid_container), len(self.gallery['img_paths']), len(self.gallery['img_paths'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'iids', 'images'])
        table.add_row(
            ['query', num_query_iids, num_query_imgs])
        table.add_row(
            ['gallery', num_gallery_iids, num_gallery_imgs])
        self.logger.info('\n' + str(table))


    def _split_anno(self, query_path: str, gallery_path: str):

        query_annos = read_json(query_path)
        gallery_annos = read_json(gallery_path)

        return query_annos, gallery_annos

    def _process_query(self, annos: List[dict]):
        pid_container = set()
        iid_container = set()
        dataset = {}
        img_paths = []
        captions = []
        person_ids = []
        instance_ids = []
        for anno in annos:
            pid = int(anno['person_id'])
            iid = int(anno['instance_id'])
            pid_container.add(pid)
            iid_container.add(iid)
            img_path = op.join(self.img_dir, anno['file_path'])
            img_paths.append(img_path)
            person_ids.append(pid)
            instance_ids.append(iid)
            captions.append(anno['caption']) # caption list

        dataset = {
            "person_ids": person_ids,
            "img_paths": img_paths,
            "instance_ids": instance_ids,
            "captions": captions
        }
        return dataset, pid_container, iid_container

    def _process_gallery(self, annos: List[dict]):
        pid_container = set()
        iid_container = set()
        dataset = {}
        img_paths = []
        person_ids = []
        instance_ids = []
        for anno in annos:
            pid = int(anno['person_id'])
            iid = int(anno['instance_id'])
            pid_container.add(pid)
            iid_container.add(iid)
            img_path = op.join(self.img_dir, anno['file_path'])
            img_paths.append(img_path)
            person_ids.append(pid)
            instance_ids.append(iid)

        dataset = {
            "person_ids": person_ids,
            "img_paths": img_paths,
            "instance_ids": instance_ids,
        }
        return dataset, pid_container, iid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))
        if not op.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
