❯ 分析这个文件夹，我现在想复用将视频直接渲染为三维手部的那些所有代码。然后我还
得考虑一个问题，就是视频跟手部渲染的同步，然后手部的渲染要放大。我可以先预先跑
完渲染的代码，把渲染结果放在数据库。把这些所有能复用代码全部给我找出来。

● Explore(探索项目中手部3D渲染相关代码)
  ⎿  Done (50 tool uses · 94.5k tokens · 18m 57s)
  (ctrl+o to expand)

  Searched for 2 patterns, read 1 file (ctrl+o to expand)

● 好的，已经完整分析了整个项目。下面是可复用代码的完整清单，以及针对你三个需求的
  分析。

---

  可复用代码清单（视频 → 3D 手部渲染）

  核心数据结构（直接复用）

  Test.cs 第17-41行
  HandLandmarkData  // 单个关键点 {id, x, y, z}
  HandData          // 单手数据 {hand_type, bound_area, hand_gesture,
  landmarks[21]}
  GestureData       // 一帧数据 {hand_count, hands[]}

  3D 手部骨骼驱动（核心渲染逻辑）

  Test.cs 第663-778行 — ProcessGestureData()

  这是最核心的可复用代码，分两步：

1. 关键点球体定位：把 21 个 landmark 坐标映射到 Unity 空间（z 轴乘以 4）
2. 骨骼驱动：用手腕/食指/中指三点算手掌朝向，再逐关节 LookAt 驱动 5
   根手指骨骼链

  关键坐标系变换（放大在这里改）：
  // 第668行，z * 4 是深度放大系数，可改为更大的值
  left_spheres[landmark.id].transform.localPosition = new Vector3(landmark.x,
  landmark.y, landmark.z * 4);

  // 第689行，手腕位置的偏移
  left_points[0].transform.localPosition = new Vector3(mark0.x / 2 - 0.15f,
  -mark0.y / 2 - 0.3f, ...);

  平滑滤波（直接复用）

  ┌────────────────────────┬─────────────────────────────────────────────────┐
  │          文件          │                      功能                       │
  ├────────────────────────┼─────────────────────────────────────────────────┤
  │ KalmanFilter.cs        │ 单维卡尔曼滤波                                  │
  ├────────────────────────┼─────────────────────────────────────────────────┤
  │ Vector3KalmanFilter.cs │ 三维向量滤波                                    │
  ├────────────────────────┼─────────────────────────────────────────────────┤
  │ HandLandmarkFilter.cs  │ 21个关键点批量滤波，参数: processNoise=0.008,   │
  │                        │ measurementNoise=0.03                           │
  └────────────────────────┴─────────────────────────────────────────────────┘

  3D 模型资源

- LeftHand.fbx / RightHand.fbx — 手部骨骼模型
- Sphere.prefab — 关键点可视化球体
- HandsDefaultMaterial.mat — 材质

  线程调度（复用）
- MainThread.cs — UnityMainThreadDispatcher，从网络线程安全地派发到主线程

---

  三个问题的解决方案

  问题1：预处理视频 → 存入数据库

  方案：写一个离线脚本（Python/服务器端），处理视频每一帧，用现有的手部检测模型
  提取 landmark，按帧号存入数据库：

  数据库表结构建议：
  video_id | frame_index | timestamp_ms | hand_type | landmarks_json |
  hand_gesture

  这样播放时按 frame_index 或 timestamp_ms
  查询对应的手部数据，不再需要实时推理。

  问题2：视频与手部渲染同步

  关键点：原系统是实时流，没有时间戳对齐。预存数据库后，同步方式：

  播放视频 → 监听 videoPlayer.time → 换算成 frame_index → 查数据库 → 驱动3D手部

  需要在 Update() 里：
  int currentFrame = Mathf.RoundToInt(videoPlayer.time * fps);
  // 查数据库/缓存 → ProcessGestureData(frameData[currentFrame])

  问题3：手部渲染放大

  修改 Test.cs 第668、730行 的 z 放大系数，以及 第689、752行
  的位置偏移和除法系数：

  // z 深度放大（原来是 * 4，改大）
  landmark.z * 8   // 放大深度

  // 手腕位置偏移（原来除以2，改小除数=放大）
  mark0.x / 1.5f   // x 方向放大
  -mark0.y / 1.5f  // y 方向放大

  或者直接对整个手部 GameObject 设置 transform.localScale，更简洁。

---

  总结

  真正需要复用的代码集中在 6 个文件：

1. Test.cs — 核心（骨骼驱动 + 数据结构）
2. KalmanFilter.cs
3. Vector3KalmanFilter.cs
4. HandLandmarkFilter.cs
5. MainThread.cs
6. Hold.cs — 如果需要交互

  加上 3 个资源文件：LeftHand.fbx、RightHand.fbx、Sphere.prefab。
