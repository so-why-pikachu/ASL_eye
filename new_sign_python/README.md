# Offline Hand 3D Pipeline (Python)

This folder contains a Python refactor for offline video-based 3D hand rendering.
It does not modify anything under `new_sign_language`.

## Files

- `offline_hand_pipeline.py`: main script
- `requirements.txt`: Python dependencies

## What it does

1. Extract hand landmarks from video frame-by-frame and store into MySQL.
2. Load stored landmarks by `frame_index`.
3. Render a 3D hand skeleton panel offline and export video.
4. Keep reusable logic aligned with Unity code:
   - `KalmanFilter.cs`
   - `Vector3KalmanFilter.cs`
   - `HandLandmarkFilter.cs`
   - coordinate scale parameters from `Test.cs` (`z_scale`, offsets, etc.)

## Install

```bash
pip install -r new_sign_python/requirements.txt
```

## Run

```bash
python new_sign_python/offline_hand_pipeline.py ^
  --mode pipeline ^
  --video "input.mp4" ^
  --mysql-host "127.0.0.1" ^
  --mysql-port 3306 ^
  --mysql-user "root" ^
  --mysql-password "123456" ^
  --mysql-database "sign_language" ^
  --output "new_sign_python/offline_render.mp4" ^
  --left-hand-model "new_sign_language/LeftHand.fbx" ^
  --right-hand-model "new_sign_language/RightHand.fbx" ^
  --sphere-prefab "new_sign_language/Sphere.prefab"
```

PowerShell single-line example:

```powershell
python new_sign_python/offline_hand_pipeline.py --mode pipeline --video "input.mp4" --mysql-host "127.0.0.1" --mysql-port 3306 --mysql-user "root" --mysql-password "123456" --mysql-database "sign_language" --output "new_sign_python/offline_render.mp4" --left-hand-model "new_sign_language/LeftHand.fbx" --right-hand-model "new_sign_language/RightHand.fbx" --sphere-prefab "new_sign_language/Sphere.prefab"
```

## Notes

- `--mode extract`: only write DB
- `--mode render`: only render from existing DB
- `--z-scale` default is `8.0` (equivalent to "render larger" than Unity's original `*4`)
- `LeftHand.fbx`, `RightHand.fbx`, `Sphere.prefab` are currently validated by path; the visual output uses landmark skeleton rendering.
- MySQL driver: `pymysql`
