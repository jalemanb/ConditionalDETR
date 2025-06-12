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
import datasets.transforms as T
import numpy as np
import torchvision.models as models
import torchvision.transforms as torchT
import torch.nn.functional as F
import random
from collections import defaultdict

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, max_classes = 3, k_templates = 3):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks = return_masks)
        self.k_templates = k_templates # Number of templates per class present in image
        self.max_classes = max_classes

        # Transform for image patches
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

        remove_imgs = []
        for idx, image_id in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            if len(anns) == 0:
                remove_imgs.append(image_id)
                continue
        self.ids = [idx for idx in self.ids if idx not in remove_imgs]

        # Build class_id -> list of dataset indices (not image_ids)
        self.class_to_indices = defaultdict(list)

        for idx, image_id in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            class_ids = set(ann['category_id'] for ann in anns)
            for class_id in class_ids:
                self.class_to_indices[class_id].append(idx)


    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            i, t = self._transforms(img, target)
            # make sure that after transforming, the target remains
            while t["labels"].shape[0] == 0 and target['labels'].shape[0] > 0:
                i, t = self._transforms(img, target)
            img, target = i, t

        present_labels = torch.unique(target["labels"]).tolist()

        # print("Present Labels Original", present_labels)

        # In this code a maximum number of classes per image is encouraged 
        # Images containing more than the alowed number of classes will remove 
        # the labels and bounding boxes from excenedent random classes
        if len(present_labels) > self.max_classes:

            # Shuffle the list for random selection
            random.shuffle(present_labels)

            # Create the two subsets
            allowed_classes = present_labels[:self.max_classes]
            left_over_classes = present_labels[self.max_classes:]

            valid_mask = torch.isin(target["labels"], torch.tensor(allowed_classes))

            # Updating & limiting the ammount of classess per image
            target["labels"] = target["labels"][valid_mask]
            target["boxes"] = target["boxes"][valid_mask]
            present_labels = allowed_classes


        templates_dict = {}

        for cls in present_labels:

            img_ids_t = self.class_to_indices[cls]

            templates = []


            for _ in range(self.k_templates):

                img_id_t = random.choice(img_ids_t)
                img_t, target_t = super(CocoDetection, self).__getitem__(img_id_t)

                target_t = {'image_id': img_id_t, 'annotations': target_t}

                w, h = img_t.size

                anno = target_t["annotations"]

                anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

                # Default Classes and Bboxes
                boxes = [obj["bbox"] for obj in anno]
                # guard against no boxes via resizing
                boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                boxes[:, 0::2].clamp_(min=0, max=w)
                boxes[:, 1::2].clamp_(min=0, max=h)
                classes = [obj["category_id"] for obj in anno]
                classes = torch.tensor(classes, dtype=torch.int64)
                keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                boxes = boxes[keep]
                classes = classes[keep]

                matching_indices = (classes == cls).nonzero(as_tuple=True)[0]

                random_idx_of_cls = random.choice(matching_indices.tolist())

                x1, y1, x2, y2 = boxes[random_idx_of_cls].int()
                patch = img_t.crop((x1.item(), y1.item(), x2.item(), y2.item()))  # Crop from PIL image
                template = self.patch_augmentation(patch)
                templates.append(template)

            templates_batch = torch.stack(templates)  # shape: [num_boxes, 3, 128, 128]
            
            templates_dict[int(cls)] = templates_batch.clone()

        return img, templates_dict, target

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


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # Default Classes and Bboxes
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

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
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])


        area = area[keep]
        iscrowd = iscrowd[keep]

        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


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
