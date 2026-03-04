import cv2
import json
import os
import random
import config_3d


def check_left_hand_reprojection():
    # 指定保存目录
    debug_out_dir = config_3d.DEBUG_OUT_DIR
    os.makedirs(debug_out_dir, exist_ok=True)

    # 1. 寻找所有包含左手(L)数据的路径
    gloss_list = os.listdir(config_3d.HAND_CROP_DIR)
    
    # 随机找一个有左手文件夹的视频
    found = False
    while not found:
        gloss = random.choice(gloss_list)
        vid_id = random.choice(os.listdir(os.path.join(config_3d.HAND_CROP_DIR, gloss)))
        crop_path_L = os.path.join(config_3d.HAND_CROP_DIR, gloss, vid_id, 'L')
        if os.path.exists(crop_path_L):
            found = True
            
    meta_path = os.path.join(crop_path_L, "meta.json")
    video_path = os.path.join(config_3d.VIDEO_DIR, f"{vid_id}.mp4")
    
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)

    # 2. 读取原视频
    cap = cv2.VideoCapture(video_path)
    test_idx = len(meta_data) // 2
    frame_meta = meta_data[test_idx]
    
    # 定位到原视频对应的帧
    orig_frame_idx = frame_meta['original_start_frame'] + frame_meta['frame_idx']
    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 3. 还原 BBox
        offset = frame_meta['offset']
        scale = frame_meta['scale']
        is_flipped = frame_meta.get('is_flipped', False)
        
        # 绘制绿色矩形框
        p1 = (int(offset[0]), int(offset[1]))
        p2 = (int(offset[0] + scale), int(offset[1] + scale))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
        
        # 标记信息
        text = f"ID: {vid_id} | Side: L | Flipped: {is_flipped}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        save_path =os.path.join(debug_out_dir, f"debug_LEFT_{vid_id}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"左手检查完成！请查看文件: {save_path}")
        print(f"验证点：红字应显示 Flipped: True，绿框应框住视频中人物的【左手】")

if __name__ == "__main__":
    check_left_hand_reprojection()