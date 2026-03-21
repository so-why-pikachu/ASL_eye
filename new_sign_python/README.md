# Backend Timestamp Tagging (Python)

This folder now provides **backend-only** processing:
- Extract hand landmarks from video
- Store frame timestamps / sync parameters into MySQL
- Store baseline hand model paths (`LeftHand.fbx`, `RightHand.fbx`) into metadata
- Export Unity-friendly playback JSON

No split-screen rendering is generated in this backend pipeline.

## Files

- `offline_hand_pipeline.py`: main backend script
- `requirements.txt`: dependencies

## Database Tables

- `video_meta`: video-level sync and render parameters, model paths
- `video_frames`: one row per frame (`frame_index`, `timestamp_ms`)
- `hand_landmarks`: hand landmarks rows aligned with frame/timestamp

## Install

```bash
pip install -r new_sign_python/requirements.txt
```

## Extract (write tags to MySQL)

```powershell
conda run -n voice_to_gemini python new_sign_python/offline_hand_pipeline.py \
  --mode extract \
  --video "F:\\sign_language\\data\\asl\\asl_first20\\0000197996356050556-CELERY.mp4" \
  --mysql-host 127.0.0.1 --mysql-port 3306 \
  --mysql-user root --mysql-password 123456 --mysql-database sign_language \
  --left-hand-model "F:\\sign_language\\new_sign_language\\LeftHand.fbx" \
  --right-hand-model "F:\\sign_language\\new_sign_language\\RightHand.fbx"
```

## Export Unity JSON (meta + frames)

```powershell
conda run -n voice_to_gemini python new_sign_python/offline_hand_pipeline.py \
  --mode export-json \
  --video-id "0000197996356050556-CELERY" \
  --mysql-host 127.0.0.1 --mysql-port 3306 \
  --mysql-user root --mysql-password 123456 --mysql-database sign_language \
  --output-json "F:\\sign_language\\new_sign_python\\unity_playback_data.json"
```

## Notes

- Timestamp/sync parameters are persisted in DB for Unity-side playback.
- Unity rendering quality (fine mesh, 360 rotation, zoom) is implemented in Unity side using these tags.

## Export Unity GestureData Stream (directly compatible shape)

Current `Test.cs` expects per-frame payload:

```json
{
  "hand_count": 1,
  "hands": [ ... ]
}
```

You can export JSONL (one frame per line):

```powershell
conda run -n voice_to_gemini python new_sign_python/offline_hand_pipeline.py \
  --mode export-gesture-stream \
  --video-id "0000197996356050556-CELERY" \
  --mysql-host 127.0.0.1 --mysql-port 3306 \
  --mysql-user root --mysql-password 123456 --mysql-database sign_language \
  --output-jsonl "F:\\sign_language\\new_sign_python\\unity_gesture_stream_0000197996356050556-CELERY.jsonl"
```

## Panda3D 3D Viewer (360 / Zoom)

You can inspect the same stream with Panda3D:

```powershell
conda run -n voice_to_gemini python new_sign_python/panda3d_hand_viewer.py `
  --stream "F:\sign_language\new_sign_python\unity_gesture_stream_0000197996356050556-CELERY.jsonl"
```

Controls:
- Left mouse drag: orbit 360 degrees
- Mouse wheel: zoom in/out
- Space: play/pause
- Left/Right arrow: step frame

## Panda3D Skin Mesh Viewer (refined)

For fuller skin-like hand surface (capsule mesh + smooth joints):

```powershell
conda run -n voice_to_gemini python new_sign_python/panda3d_skin_mesh_viewer.py `
  --stream "F:\sign_language\new_sign_python\unity_gesture_stream_0000197996356050556-CELERY.jsonl"
```

This mode is designed for more detailed visual quality and supports 360 orbit + zoom.

## Panda3D FBX Skinning Viewer (true model-driven)

Uses your real hand models:
- `new_sign_language/LeftHand.fbx`
- `new_sign_language/RightHand.fbx`

and drives FBX skeleton joints from landmark stream:

```powershell
conda run -n voice_to_gemini python new_sign_python/panda3d_fbx_skinning_viewer.py `
  --stream "F:\sign_language\new_sign_python\unity_gesture_stream_0000197996356050556-CELERY.jsonl" `
  --left-fbx "F:\sign_language\new_sign_language\LeftHand.fbx" `
  --right-fbx "F:\sign_language\new_sign_language\RightHand.fbx" `
  --pos-smooth 0.72 `
  --rot-smooth 0.82 `
  --wrist-stability 0.65 `
  --offset-config "F:\sign_language\new_sign_python\fbx_bone_offsets.example.json"
```

Quality tuning:
- Increase `--rot-smooth` (e.g. `0.88`) to reduce finger jitter.
- Increase `--pos-smooth` (e.g. `0.80`) for steadier hand root motion.
- Increase `--wrist-stability` (e.g. `0.75`) to suppress palm/wrist wobble.
- Use `--offset-config` to fine tune per-bone HPR offsets for best realism.

Auto-calibrate offsets from your stream:

```powershell
conda run -n voice_to_gemini python new_sign_python/panda3d_fbx_skinning_viewer.py `
  --stream "F:\sign_language\new_sign_python\unity_gesture_stream_0000197996356050556-CELERY.jsonl" `
  --left-fbx "F:\sign_language\new_sign_language\LeftHand.fbx" `
  --right-fbx "F:\sign_language\new_sign_language\RightHand.fbx" `
  --calibrate-output "F:\sign_language\new_sign_python\fbx_bone_offsets.auto.json" `
  --calibrate-max-frames 600 `
  --calibrate-strength 0.45
```
