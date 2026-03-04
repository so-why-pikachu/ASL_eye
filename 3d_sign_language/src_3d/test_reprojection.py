import cv2
import json
import os
import numpy as np
import random
import sys
import config_3d

def visualize_reprojection(num_samples=1):
    # 指定保存目录
    debug_out_dir = config_3d.DEBUG_OUT_DIR
    os.makedirs(debug_out_dir, exist_ok=True)

    # 1. 随机选择一个处理好的视频
    gloss_list = os.listdir(config_3d.HAND_CROP_DIR)
    gloss = random.choice(gloss_list)
    vid_id = random.choice(os.listdir(os.path.join(config_3d.HAND_CROP_DIR, gloss)))
    side = random.choice(['R', 'L'])
    
    crop_path = os.path.join(config_3d.HAND_CROP_DIR, gloss, vid_id, side)
    meta_path = os.path.join(crop_path, "meta.json")
    video_path = os.path.join(config_3d.VIDEO_DIR, f"{vid_id}.mp4")
    
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)

    # 2. 读取原视频
    cap = cv2.VideoCapture(video_path)
    
    # 3. 抽取中间的一帧进行检查
    test_idx = len(meta_data) // 2
    frame_meta = meta_data[test_idx]
    
    # 定位到原视频对应的帧
    orig_frame_idx = frame_meta['original_start_frame'] + frame_meta['frame_idx']
    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 4. 根据 meta 信息还原 BBox 矩形
        offset = frame_meta['offset']
        scale = frame_meta['scale']
        
        # 画出还原后的绿色矩形框
        p1 = (int(offset[0]), int(offset[1]))
        p2 = (int(offset[0] + scale), int(offset[1] + scale))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
        
        # 5. 显示结果
        cv2.putText(frame, f"ID: {vid_id} Side: {side}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存或展示
        save_path = os.path.join(debug_out_dir, f"debug_{vid_id}_{side}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Check completed! Saved to {save_path}")

if __name__ == "__main__":
    visualize_reprojection()