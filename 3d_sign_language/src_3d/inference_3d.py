import torch
import numpy as np
import os
import json
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import sys

# 确保导入路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src_3d import config_3d
    # 假设你的 S2HAND 模型类定义在 src_3d.model 中
    from src_3d.model import S2HAND 
except ImportError:
    import config_3d
    from model import S2HAND

# 1. 基础配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # 4060 建议设置 32-64
SAVE_DIR = os.path.join(os.path.dirname(config_3d.HAND_CROP_DIR), "hand_3d_features")
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(checkpoint_path):
    """加载 S2HAND 模型"""
    model = S2HAND().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

@torch.no_grad()
def process_video_folder(model, video_path, side, is_flipped):
    """处理单个视频文件夹（R或L）"""
    img_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
    if not img_files:
        return None

    # 加载所有帧并堆叠成 Batch
    imgs = []
    for f in img_files:
        img = Image.open(os.path.join(video_path, f)).convert('RGB')
        imgs.append(transform(img))
    
    img_tensor = torch.stack(imgs).to(DEVICE)
    
    # 模型推理
    # 强制 shape_params (betas) 为 0 以获得标准手动作
    outputs = model(img_tensor, betas=torch.zeros((len(img_files), 10)).to(DEVICE))
    
    # 提取核心参数 (转换为 numpy)
    joints_3d = outputs['joints_3d'].cpu().numpy() # (N, 21, 3)
    pose_params = outputs['pose_params'].cpu().numpy() # (N, 48)
    cam_params = outputs['cam_params'].cpu().numpy() # (N, 3) -> [scale, tx, ty]
    
    # 【核心逻辑】左手镜像还原
    if is_flipped:
        # 1. 3D 关节坐标：X轴取反
        joints_3d[:, :, 0] = -joints_3d[:, :, 0]
        
        # 2. Camera 参数：横向偏移 tx 取反 (在图像坐标系下)
        cam_params[:, 1] = -cam_params[:, 1]
        
        # 3. Pose 参数：MANO 旋转参数的镜像处理较为复杂，
        # 如果你后续使用关节坐标重建，坐标的反转已经足够。
        # 这里保留原始 theta，由 3D 引擎根据侧边标签(Side)自动映射。

    return {
        "joints": joints_3d,
        "pose": pose_params,
        "camera": cam_params,
        "frames": img_files
    }

def main():
    # 加载模型 (请确保路径正确)
    model = load_model("checkpoints/s2hand_best.pth")
    
    # 遍历预处理好的 4万个视频
    gloss_list = os.listdir(config_3d.HAND_CROP_DIR)
    pbar = tqdm(gloss_list, desc="Processing 3D Reconstruction")

    for gloss in pbar:
        gloss_path = os.path.join(config_3d.HAND_CROP_DIR, gloss)
        for vid_id in os.listdir(gloss_path):
            vid_path = os.path.join(gloss_path, vid_id)
            
            for side in ['R', 'L']:
                side_path = os.path.join(vid_path, side)
                if not os.path.exists(side_path): continue
                
                # 读取 meta.json 获取翻转状态
                with open(os.path.join(side_path, "meta.json"), 'r') as f:
                    meta = json.load(f)[0]
                
                # 执行推理
                result = process_video_folder(model, side_path, side, meta['is_flipped'])
                
                if result:
                    # 保存结果为压缩文件，节省空间
                    out_vid_dir = os.path.join(SAVE_DIR, gloss, vid_id)
                    os.makedirs(out_vid_dir, exist_ok=True)
                    save_path = os.path.join(out_vid_dir, f"features_{side}.npz")
                    
                    np.savez_compressed(
                        save_path,
                        joints=result['joints'],
                        pose=result['pose'],
                        camera=result['camera'],
                        is_flipped=meta['is_flipped']
                    )

if __name__ == "__main__":
    main()