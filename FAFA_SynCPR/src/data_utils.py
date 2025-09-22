import json
from pathlib import Path
from typing import Union, List, Dict, Literal

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, Pad
import torch 
import os.path as op
from prettytable import PrettyTable
from PIL import Image, ImageFile

PIL.Image.MAX_IMAGE_PIXELS = None

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not op.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TrainResizeRandomCrop:
    """
    Resize the image to dim + 10, then random crop it to dim.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, image):
        # Resize the image to dim + 10
        image = Resize((self.dim + 10, self.dim + 10), interpolation=PIL.Image.BICUBIC)(image)
        # Random crop it to dim
        image = RandomCrop(self.dim)(image)
        return image


class InferenceResize:
    """
    Resize the image to dim for inference.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, image):
        # Resize the image to dim for inference
        image = Resize((self.dim, self.dim), interpolation=PIL.Image.BICUBIC)(image)
        return image


def train_transform(dim: int):
    """
    Training transform: Resize to dim + 10, random crop to dim, random horizontal flip, and then normalize the image.
    """
    return Compose([
        TrainResizeRandomCrop(dim),
        RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def inference_transform(dim: int):
    """
    Inference transform: Resize to dim and then normalize the image.
    """
    return Compose([
        InferenceResize(dim),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def squarepad_transform_test(dim: int, need_size=(384, 192)):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        Resize(need_size, interpolation=PIL.Image.BICUBIC),
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def squarepad_transform_train(dim: int, need_size=(384, 192)):
    """
    Augmented CLIP-like preprocessing transform:
    1. Random horizontal flip
    2. Pad 10 pixels
    3. Random crop back to original size
    4. SquarePad
    5. Standard preprocessing (resize, crop, normalize)
    :param dim: image output dimension
    :return: torchvision Compose transform
    """
    return Compose([
        RandomHorizontalFlip(0.5),
        Pad(10),
        RandomCrop((need_size[0],need_size[1])),  # crop back to original image size
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        lambda img: img.convert('RGB'),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class SynCPRDataset(Dataset):
    """
    Synthetic Composed Person Retrieval Dataset (SynCPR)
    Large-scale synthetic dataset for training FAFA model
    """

    def __init__(self, data_path: Union[str, Path]='/your/custom/syncpr/root', json_path="SynCPR.json", split: Literal['train', 'val', 'test']='train',
                 mode: Literal['relative', 'classic']='relative', preprocess: callable=None, setting: List[str]=['norm', 'change']):
        """
        Args:
            data_path (Union[str, Path]): path to CIRHS dataset
            split (str): dataset split, should be in ['train', 'test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
            setting (List[str]): setting to cover different semantic changes
        """
        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['train', 'test', 'val']:
            raise ValueError("split should be in ['train', 'test', 'val']")

        # Load Annotation Information
        
        with open(op.join(data_path, json_path), 'r', encoding='utf-8') as file:
            self.annotations = json.load(file)
        # for item in setting:
        #     try:
        #         with open(data_path / 'CIR_data' / item / "processed_data_bi_v2.json", "r") as f:
        #             self.annotations.extend(json.load(f))
        #     except:
        #         raise ValueError(f"The setting {item} is not implemented yet")

        if len(self.annotations) == 0:
            raise IOError("The training data contains noting")

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23
        print(f"SynCPRDataset {split} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.
        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """
        try:
            if self.mode == 'relative':
                # Get the query id
                cpr_id = self.annotations[index]['cpr_id']

                # Get relative caption and shared concept
                relative_caption = self.annotations[index]['edit_caption']

                # Get the reference image
                reference_img_path = self.data_path / self.annotations[index]['reference_image_path']
                reference_img = self.preprocess(PIL.Image.open(reference_img_path))

                if self.split in ['val', 'test']:
                    raise ValueError("The split 'val' has not been implemented yet")
                else:
                    # Get the target image and ground truth images
                    target_img_path = self.data_path / self.annotations[index]['target_image_path']
                    target_img = self.preprocess(PIL.Image.open(target_img_path))

                    return reference_img, target_img, relative_caption, cpr_id
            else:
                raise ValueError("The mode 'classic' has not been implemented yet")
        
        except Exception as e:
                print(f"Exception: {e}")

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        else:
            raise ValueError("The mode 'classic' has not been implemented yet")


class ITCPRDataset(Dataset):
    """
    In-the-wild Test dataset for Composed Person Retrieval (ITCPR)
    Manually annotated test set for evaluating CPR models

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html
    """
    dataset_dir = 'datasets'

    def __init__(self, root='/your/custom/itcpr/root', verbose=True):
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
            print("=> ITCPR Images and Captions are loaded")
            self.show_itcpr_info()
    
    def show_itcpr_info(self):
        num_query_pids, num_query_iids, num_query_imgs = len(
            self.query_pid_container), len(self.query_iid_container), len(self.query['img_paths'])
        num_gallery_pids, num_gallery_iids, num_gallery_imgs = len(
            self.gallery_pid_container), len(self.gallery['img_paths']), len(self.gallery['img_paths'])

        # TODO use prettytable print comand line table

        print(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'iids', 'images'])
        table.add_row(
            ['query', num_query_iids, num_query_imgs])
        table.add_row(
            ['gallery', num_gallery_iids, num_gallery_imgs])
        print('\n' + str(table))


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


class GalleryDataset(Dataset):
    def __init__(self, instance_ids, img_paths, preprocess: callable=None):
        self.instance_ids = instance_ids
        self.img_paths = img_paths
        self.transform = preprocess

    def __len__(self):
        return len(self.instance_ids)

    def __getitem__(self, index):
        iid, img_path = self.instance_ids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return iid, img

class QueryDataset(Dataset):
    def __init__(self, instance_ids, img_paths, captions, preprocess: callable=None):
        self.instance_ids = instance_ids
        self.img_paths = img_paths
        self.transform = preprocess
        self.captions = captions

    def __len__(self):
        return len(self.instance_ids)

    def __getitem__(self, index):
        iid, img_path, caption= self.instance_ids[index], self.img_paths[index], self.captions[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return iid, img, caption