import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomObjectDetectionDataset(Dataset):
    """
    Custom Dataset class cho bài toán Object Detection.
    Đọc dữ liệu ảnh và bounding box từ định dạng JSON (tương tự COCO).
    """
    def __init__(self, data_dir, ann_file, transforms=None):
        """
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa ảnh.
            ann_file (str): Đường dẫn đến file annotation (JSON).
            transforms (albumentations.Compose): Các phép biến đổi dữ liệu.
        """
        self.data_dir = data_dir
        self.transforms = transforms
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
        # Nhóm annotations theo image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image bằng OpenCV
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh tại: {img_path}. Vui lòng kiểm tra lại tệp tin.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển sang RGB
        
        # Load bounding boxes và labels
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        area = []
        iscrowd = []
        
        for ann in anns:
            # COCO format: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            # Chuyển sang format Pascal VOC: [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            area.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
            
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Áp dụng Data Augmentation (Resize, Flip, Rotation, Blur, Color Jitter)
        if self.transforms:
            # Albumentations yêu cầu bboxes phải là list/array
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['class_labels'], dtype=np.int64)

        # Xử lý trường hợp không có box nào sau khi crop/transform
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Format target cho PyTorch Object Detection
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

def get_transforms(train=True):
    """
    Hàm tạo pipeline Data Augmentation bằng Albumentations.
    """
    if train:
        return A.Compose([
            A.Resize(800, 800), # Resize ảnh chuẩn
            A.HorizontalFlip(p=0.5), # Random Flip
            A.Rotate(limit=15, p=0.3), # Rotation nhẹ
            A.GaussianBlur(blur_limit=(3, 7), p=0.2), # Blur
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3), # Color Jitter
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize (ImageNet)
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=10.0, min_visibility=0.1))
    else:
        return A.Compose([
            A.Resize(800, 800),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def collate_fn(batch):
    """
    Hàm xử lý batch cho DataLoader (vì object detection có số lượng boxes khác nhau mỗi ảnh)
    """
    return tuple(zip(*batch))
