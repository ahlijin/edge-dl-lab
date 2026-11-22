import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class HomeObjectsDataset(Dataset):
    def __init__(self, data_dir, split="train", input_size=(320, 320)):
        self.img_dir = os.path.join(data_dir, "images", split)
        self.label_dir = os.path.join(data_dir, "labels", split)
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        self.input_size = input_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 加载图像并resize
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize(self.input_size)
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(self.input_size[1], self.input_size[0], 3).permute(2, 0, 1) / 255.0

        # 解析YOLO标签
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        # YOLO格式：x_center, y_center, w, h（全部归一化到0-1）
                        cx, cy, w, h = map(float, parts[1:])

                        # 转换为图像绝对像素坐标(cx, cy, w, h)
                        abs_cx = cx * self.input_size[0]
                        abs_cy = cy * self.input_size[1]
                        abs_w = w * self.input_size[0]
                        abs_h = h * self.input_size[1]

                        boxes.append([abs_cx, abs_cy, abs_w, abs_h])
                        classes.append(cls_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)
        return img_tensor, boxes, classes