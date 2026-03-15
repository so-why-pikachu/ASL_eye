接口一：模型推理接口
路由： POST /api/sign/predict
Content-Type： multipart/form-data
Request Body：
字段类型必填说明videoFile✅用户录制的手语片段（mp4 / webm / avi）
后端处理流程（对应你的代码）：

用 OpenCV VideoCapture 逐帧解码上传的视频文件
每帧调用 pipeline.extract_features(frame) → 提取 134 维坐标
每帧经过 pipeline.smoother.smooth(raw_feat) → 平滑处理
攒成 frame_buffer 后调用 pipeline.predict(frame_buffer)
内部走 preprocess_sequence → 双重相对坐标 + 线性抽样 → 模型推理

Response Body：
json{
  "code": 200,
  "data": {
    "word_name": "ACCIDENT",
    "confidence": 0.98
  }
}
错误响应：
json{
  "code": 400,
  "message": "Video too short, minimum 5 frames required"
}

对应代码中 preprocess_sequence 的 if len(raw_seq) < 5: return None 分支。


接口二：资源索引接口
路由： GET /api/sign/downloads
Query Params： name=ACCIDENT
功能： 查 MySQL sign_assets 表 → 找到物理文件夹 → 扫描并返回 Web URLs。