# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import selective_search
import datasets.transforms as T
import numpy as np
import torchvision.models as models
import torchvision.transforms as torchT
import torch.nn.functional as F
import random

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}


        img, templates, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, templates, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def generate_random_boxes(k, image_width, image_height, min_aspect_ratio=0.3):
    """
    Generate k valid random bounding boxes (x1, y1, x2, y2) within image bounds,
    each with a minimum aspect ratio constraint.

    Args:
        k (int): Number of boxes
        image_width (int): Width of the image
        image_height (int): Height of the image
        min_aspect_ratio (float): Minimum allowed aspect ratio (w/h or h/w)

    Returns:
        boxes (torch.Tensor): shape [k, 4]
        areas (torch.Tensor): shape [k]
        iscrowd (torch.Tensor): shape [k, 1]
    """
    boxes = []
    max_tries = 50 * k  # Try more if aspect ratio filtering is involved

    k = random.randint(2, k)

    tries = 0
    while len(boxes) < k and tries < max_tries:
        tries += 1

        # Random top-left corner
        x1 = torch.randint(0, image_width - 1, (1,)).item()
        y1 = torch.randint(0, image_height - 1, (1,)).item()

        # Random width and height
        max_w = image_width - x1
        max_h = image_height - y1
        if max_w <= 1 or max_h <= 1:
            continue

        w = torch.randint(1, max_w, (1,)).item()
        h = torch.randint(1, max_h, (1,)).item()

        # Check aspect ratio
        aspect = max(w / h, h / w)  # force it to be >= 1
        if aspect < 1 / min_aspect_ratio:
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1, y1, x2, y2])

    if len(boxes) < k:
        print(f"Warning: Only generated {len(boxes)} boxes after {tries} tries.")

    boxes = torch.tensor(boxes, dtype=torch.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iscrowd = torch.zeros((boxes.shape[0], 1), dtype=torch.int64)

    return boxes, areas, iscrowd



class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.eval()
        # Transform for image patches
        self.patch_transform = torchT.Compose([
            torchT.Resize((224, 224)),
            torchT.ToTensor(),
            torchT.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.patch_augmentation = torchT.Compose([
            torchT.Resize((128, 128)),  # Tight crop is assumed already
            torchT.RandomApply([
                torchT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            torchT.RandomGrayscale(p=0.2),
            torchT.RandomApply([torchT.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
            torchT.ToTensor(),
            torchT.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # Default Classes and Bboxes
        # boxes = [obj["bbox"] for obj in anno]
        # # guard against no boxes via resizing
        # boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes[:, 2:] += boxes[:, :2]
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)
        # classes = [obj["category_id"] for obj in anno]
        # classes = torch.tensor(classes, dtype=torch.int64)

        # Get Random Crops in the image
        boxes, area, iscrowd = generate_random_boxes(10, w, h)
        
        #  Extract the patches and normalize them to create the binary labels object/no object
        patches = []
        templates = []
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            patch = image.crop((x1.item(), y1.item(), x2.item(), y2.item()))  # Crop from PIL image
            patch_class = self.patch_transform(patch)
            template = self.patch_augmentation(patch)
            patches.append(patch_class)
            templates.append(template)

        patch_batch = None
        if patches:
            patch_batch = torch.stack(patches)  # shape: [num_boxes, 3, 224, 224]
            templates_batch = torch.stack(templates)  # shape: [num_boxes, 3, 128, 128]

            with torch.no_grad():
                outputs = self.resnet18(patch_batch)
                probs = F.softmax(outputs, dim=1)
                has_object = (probs.max(dim=1).values > 0.2).long()  # 1 if any class > threshold
                classes = has_object.unsqueeze(1)  # shape: [num_boxes, 1]


        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]


        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, templates_batch, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images" / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "images" / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "images" / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
