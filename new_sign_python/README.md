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
