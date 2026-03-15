## 1. 业务逻辑流

1. **特征提交** ：前端通过 `predict` 接口发送关键点序列。
2. **AI 推理** ：后端加载 `.pth` 跑出 `label_id`，并通过 `ID_TO_NAME` 映射为词条（如 `ACCIDENT`）。
3. **资源检索** ：前端拿着词条名请求 `downloads` 接口。
4. **路径解析** ：后端查询 MySQL，根据词条找到对应的复杂文件夹名（如 `ACCIDENT_5977...`），递归扫描磁盘，返回所有 `.glb` 的 Web URL。

---

## 2. 接口详细定义

### 接口一：模型推理接口

* **路由** : `POST /api/sign/predict`
* **功能** : 接收手语特征，输出识别结果。
* **Request Body (JSON)** :

```
  {
    "features": [
      [x1, y1, x2, y2, "..."], // 第1帧的134维特征
      [x1, y1, x2, y2, "..."]  // 第n帧的134维特征
    ]
  }
```

* **Response Body (JSON)** :

```
  {
    "code": 200,
    "msg": "success",
    "data": {
      "label_id": 5,
      "word_name": "ACCIDENT",
      "confidence": 0.96
    }
  }
```

---

### 接口二：资源索引接口

* **路由** : `GET /api/sign/downloads`
* **功能** : 根据词条名，返回其对应的所有 3D 帧文件链接。
* **Query Params** : `name=ACCIDENT`
* **Response Body (JSON)** :

```
  {
    "code": 200,
    "msg": "success",
    "data": {
      "word": "ACCIDENT",
      "total_frames": 60,
      "model_urls": [
        "http://server:5000/static/models/ACCIDENT_5977.../frame_000.glb",
        "http://server:5000/static/models/ACCIDENT_5977.../frame_001.glb",
        "..."
      ]
    }
  }
```

---

### 接口三：静态资源服务

* **路由** : `GET /static/models/<path:file_path>`
* **功能** : 实际传输 `.glb` 二进制流，支持浏览器缓存和 Three.js 加载。

---

## 3. 数据库表结构设计 (MySQL)

你必须建立这张表，否则后端无法从简洁的词语 `ACCIDENT` 找到磁盘上那个带有随机长 ID 的文件夹。

### 表名: sign_language_db

| **字段名** | **类型** | **约束** | **说明**                 |
| ---------------- | -------------- | -------------- | ------------------------------ |
| `id`           | INT            | PRIMARY KEY    | 自增 ID                        |
| `word_name`    | VARCHAR(50)    | UNIQUE, INDEX  | 词语名，对应 `predict`的输出 |
| `folder_name`  | VARCHAR(255)   | NOT NULL       | 磁盘上真实的文件夹全名         |
