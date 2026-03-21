using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Android;
using System.Linq;
using Unity.XR.XREAL;
using Unity.XR.XREAL.Samples;
using Unity.VisualScripting;

[System.Serializable]
public class HandLandmarkData
{
    public int id;
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class HandData
{
    public int hand_index;
    public string hand_type;
    public float bound_area;
    public string hand_gesture;
    public HandLandmarkData[] landmarks;
}

[System.Serializable]
public class GestureData
{
    public int hand_count;
    public HandData[] hands;
}

public class Test : MonoBehaviour
{
    public GameObject left_hand;
    public GameObject right_hand;
    public GameObject target;
    public Hold left_hold;
    public Hold right_hold;
    public Text left_result;
    public Text right_result;

    // 缩放操纵
    public Vector3 LastLeftHandPosition;
    public Vector3 LastRightHandPosition;
    public Quaternion LastLeftHandRotation;
    public Quaternion LastRightHandRotation;
    private float lastHandDistance;

    public List<GameObject> left_points = new List<GameObject>();
    public List<GameObject> right_points = new List<GameObject>();
    public List<GameObject> left_spheres = new List<GameObject>(); 
    public List<GameObject> right_spheres = new List<GameObject>(); 
    public GameObject sphere;

    private List<Vector3> origin_leftRotations = new List<Vector3>();
    private List<Vector3> origin_rightRotations = new List<Vector3>();

    // 手势滤波：滑动窗口多数表决
    [SerializeField] private int gestureWindowSize = 8; // 窗口大小（可调，3-7较合适）
    private Queue<string> leftGestureHistory = new Queue<string>(); // 左手手势历史
    private Queue<string> rightGestureHistory = new Queue<string>(); // 右手手势历史
    private string filteredLeftGesture = ""; // 过滤后的左手手势
    private string filteredRightGesture = ""; // 过滤后的右手手势

    // 卡尔曼滤波
    private HandLandmarkFilter landmarkFilter;

    // 网络配置
    [SerializeField] private string serverIP = "117.50.46.14";
    [SerializeField] private int serverPort = 8080;

    // 摄像头和网络相关
    private WebCamTexture webcamTexture;
    public RGBCameraExample arCamera;
    public RawImage arCameraImage;
    public RenderTexture arCameraTexture;
    public ComputeShader YUV2RGBShader;
    private TcpClient client;
    private NetworkStream stream;
    private Thread sendThread;
    private bool isRunning = false;
    private bool disconnecting = false;

    // 帧数据队列（线程安全）
    private readonly Queue<byte[]> frameQueue = new Queue<byte[]>();
    private readonly object queueLock = new object();

    // 图像压缩质量
    [Range(10, 100)]
    [SerializeField] private int jpegQuality = 100;

    // 帧处理间隔（毫秒）
    [SerializeField] private int frameInterval = 33;
    private float lastFrameTime = 0;

    // 标记是否需要捕获帧
    private bool captureThisFrame = false;
    private Texture2D texture;

    [SerializeField] private Texture2D testImage;

    // 权限请求相关
    private bool permissionGranted = false;
    private bool permissionRequested = false;

    // 新增：接收线程和相关变量
    private Thread receiveThread;
    private bool isReceiving = false;

    void Start()
    {
        // 检查并请求摄像头权限
        CheckCameraPermission();

        // 确保主线程调度器存在
        if (UnityMainThreadDispatcher.Instance == null)
        {
            new GameObject("UnityMainThreadDispatcher").AddComponent<UnityMainThreadDispatcher>();
        }

        // 初始化滤波器，可调整噪声参数
        landmarkFilter = new HandLandmarkFilter(0.008f, 0.03f);

        for(int i = 0;i<left_points.Count;i++)
        {
            origin_leftRotations.Add(left_points[i].transform.localEulerAngles);
            origin_rightRotations.Add(right_points[i].transform.localEulerAngles);
        }
    }

    void CheckCameraPermission()
    {
        // 检查是否有摄像头权限
        if (Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            permissionGranted = true;
            InitializeWebcam();
            ConnectToServer();
        }
        else
        {
            // 请求摄像头权限
            if (!permissionRequested)
            {
                permissionRequested = true;
                Permission.RequestUserPermission(Permission.Camera);
            }
        }
    }

    void Update()
    {
        // 检查权限请求结果
        if (!permissionGranted && permissionRequested)
        {
            if (Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                permissionGranted = true;
                InitializeWebcam();
                ConnectToServer();
            }
        }

        // 仅在主线程中处理摄像头和纹理
        if (isRunning && !disconnecting)
        {
            // 控制帧率
            if (Time.time - lastFrameTime >= frameInterval / 1000f)
            {
                lastFrameTime = Time.time;

                // 标记需要捕获当前帧
                captureThisFrame = true;
            }
        }

        if (captureThisFrame && texture != null)
        {
            captureThisFrame = false;
            CaptureFrame();
        }
    }

    void InitializeWebcam()
    {
        // 在Android上，需要确保在权限获取后执行
        if (!permissionGranted)
        {
            return;
        }

        // 获取可用摄像头列表
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            return;
        }

        // 使用第一个摄像头
        webcamTexture = new WebCamTexture(devices[0].name, 800, 400, 30);
        webcamTexture.Play();

        // 创建一次Texture2D实例
        texture = new Texture2D(arCameraImage.texture.width, arCameraImage.texture.height, ((Texture2D)arCameraImage.texture).format, true);

        Debug.Log(texture.width + " " + texture.height);
    }

    void CaptureFrame()
    {
        //Debug.Log("capture");
        //byte[] frameData = testImage.EncodeToJPG(jpegQuality);
        // 编码为JPEG（必须在主线程）
        if (arCamera.y_texture == null) { return; }

        int width = arCamera.y_texture.width;
        int height = arCamera.y_texture.height;
        //int width = 640;
        //int height = 480;

        // 确保我们有一个合适的Texture2D用于捕获
        // 确保我们有一个合适的RenderTexture用于Compute Shader输出
        if (arCameraTexture == null ||
            arCameraTexture.width != width ||
            arCameraTexture.height != height)
        {
            // 释放旧纹理
            if (arCameraTexture != null)
            {
                RenderTexture.ReleaseTemporary(arCameraTexture);
            }

            // 创建支持UAV的渲染纹理（关键修改）
            arCameraTexture = RenderTexture.GetTemporary(
                width,
                height,
                0,
                RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Linear
            );
            arCameraTexture.enableRandomWrite = true; // 允许随机写入
            //arCameraTexture.creationFlags = RenderTextureCreationFlags.UnorderedAccess; // 启用UAV
        }

        int kernelHandle = YUV2RGBShader.FindKernel("CSMain");

        YUV2RGBShader.SetTexture(kernelHandle, "YTexture", arCamera.y_texture);
        YUV2RGBShader.SetTexture(kernelHandle, "UTexture", arCamera.u_texture);
        YUV2RGBShader.SetTexture(kernelHandle, "VTexture", arCamera.v_texture);
        YUV2RGBShader.SetTexture(kernelHandle, "Result", arCameraTexture);

        // 计算线程组数量 (8x8线程每组)
        int threadGroupsX = Mathf.CeilToInt(width / 8f);
        int threadGroupsY = Mathf.CeilToInt(height / 8f);

        // 执行Compute Shader
        YUV2RGBShader.Dispatch(kernelHandle, threadGroupsX, threadGroupsY, 1);

        // 创建临时Texture2D用于读取像素
        Texture2D tempTexture = new Texture2D(arCameraTexture.width, arCameraTexture.height, TextureFormat.RGBA32, false);

        // 激活RenderTexture以便读取
        RenderTexture.active = arCameraTexture;

        // 读取像素到Texture2D
        tempTexture.ReadPixels(new Rect(0, 0, arCameraTexture.width, arCameraTexture.height), 0, 0);
        tempTexture.Apply();

        // 恢复RenderTexture状态
        RenderTexture.active = null;

        // 编码为JPG
        byte[] frameData = tempTexture.EncodeToJPG();

        Destroy(tempTexture); // 销毁 Texture2D（必须在主线程调用）
        tempTexture = null;

        // 将帧数据添加到队列（线程安全）
        if (frameData != null && frameData.Length > 0)
        {
            lock (queueLock)
            {
                // 限制队列大小，防止内存溢出
                while (frameQueue.Count > 3)
                {
                    frameQueue.Dequeue();
                }

                frameQueue.Enqueue(frameData);
            }
        }
        try
        {
           
        }
        catch (Exception e)
        {
            Debug.LogError("捕获帧时出错: " + e.Message);
        }
    }

    void ConnectToServer()
    {
        try
        {
            // 创建TCP客户端并连接到服务器
            client = new TcpClient();
            IAsyncResult result = client.BeginConnect(serverIP, serverPort, null, null);

            // 设置连接超时时间（5秒）
            bool success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(5));

            if (client.Connected)
            {
                client.EndConnect(result);
                stream = client.GetStream();
                isRunning = true;

                // 启动发送线程
                sendThread = new Thread(SendFrames);
                sendThread.IsBackground = true; // 设置为后台线程
                sendThread.Start();

                // 新增：启动接收线程
                isReceiving = true;
                receiveThread = new Thread(ReceiveGestureData);
                receiveThread.IsBackground = true;
                receiveThread.Start();
            }
            else
            {
                client.Close();
            }
        }
        catch (Exception e)
        {

        }
    }

    void SendFrames()
    {
        try
        {
            while (isRunning && !disconnecting)
            {
                try
                {
                    // 检查客户端连接状态
                    if (client == null || !client.Connected)
                    {
                        Debug.LogWarning("服务器连接已断开");
                        break;
                    }

                    // 从队列中获取帧数据（线程安全）
                    byte[] frameData = null;
                    lock (queueLock)
                    {
                        if (frameQueue.Count > 0)
                        {
                            frameData = frameQueue.Dequeue();
                        }
                    }

                    // 发送帧数据
                    if (frameData != null && frameData.Length > 0)
                    {
                        // 发送帧数据长度 (4字节整数)
                        byte[] lengthBytes = BitConverter.GetBytes(frameData.Length);

                        if (stream != null && client.Connected)
                        {
                            stream.Write(lengthBytes, 0, lengthBytes.Length);
                            stream.Write(frameData, 0, frameData.Length);
                        }
                    }
                    else
                    {
                        // 没有帧数据时，短暂休眠
                        Thread.Sleep(10);
                    }
                }
                catch (Exception e)
                {
                    if (!disconnecting)
                    {
                        Debug.LogError("发送帧时出错: " + e.Message);
                    }
                    break;
                }
            }
        }
        finally
        {
            if (!disconnecting)
            {
                Debug.Log("发送线程已退出");
                Disconnect();
            }
        }
    }

    // 新增：接收手势数据的线程函数
    void ReceiveGestureData()
    {
        try
        {
            while (isReceiving && !disconnecting)
            {
                if (stream == null || !client.Connected)
                {
                    Debug.LogWarning("连接已断开，无法接收数据");
                    break;
                }

                // 先接收数据长度（4字节）
                byte[] lengthBytes = new byte[4];
                int bytesRead = 0;
                try
                {
                    bytesRead = stream.Read(lengthBytes, 0, 4);
                }
                catch (Exception e)
                {
                    Debug.LogError("读取数据长度时出错: " + e.Message);
                    break;
                }

                if (bytesRead != 4)
                {
                    Debug.LogWarning("接收数据长度失败，可能连接已断开");
                    break;
                }

                int dataLength = BitConverter.ToInt32(lengthBytes, 0);

                // 接收实际数据
                byte[] dataBytes = new byte[dataLength];
                int totalRead = 0;
                while (totalRead < dataLength)
                {
                    try
                    {
                        int read = stream.Read(dataBytes, totalRead, dataLength - totalRead);
                        if (read == 0)
                        {
                            Debug.LogWarning("接收数据失败，连接可能已断开");
                            break;
                        }
                        totalRead += read;
                    }
                    catch (Exception e)
                    {
                        Debug.LogError("读取数据内容时出错: " + e.Message);
                        break;
                    }
                }

                if (totalRead == dataLength)
                {
                    // 转换为字符串
                    string jsonData = Encoding.UTF8.GetString(dataBytes);

                    // 在主线程中打印
                    UnityMainThreadDispatcher.Instance.Enqueue(() =>
                    {
                        // 反序列化 JSON 数据
                        GestureData gestureData = JsonUtility.FromJson<GestureData>(jsonData);

                        // 处理手势数据
                        ProcessGestureData(gestureData);
                    });
                }
                else
                {
                    // 数据接收不完整
                    Debug.LogWarning($"数据接收不完整，期望 {dataLength} 字节，实际接收 {totalRead} 字节");
                }
            }
        }
        catch (Exception e)
        {
            if (!disconnecting)
            {
                Debug.LogError("接收手势数据出错: " + e.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        Disconnect();
    }

    void OnDestroy()
    {
        Disconnect();
    }

    void Disconnect()
    {
        // 防止重复调用
        if (disconnecting) return;

        disconnecting = true;
        isRunning = false;
        isReceiving = false; // 新增：停止接收线程

        try
        {
            Debug.Log("开始断开连接...");

            // 清空帧队列
            lock (queueLock)
            {
                frameQueue.Clear();
            }

            // 关闭网络流和客户端（子线程操作）
            if (stream != null)
            {
                try { stream.Close(); } catch { }
                stream = null;
            }

            if (client != null)
            {
                try { client.Close(); } catch { }
                client = null;
            }

            // 在主线程中停止摄像头
            if (webcamTexture != null)
            {
                webcamTexture.Stop();
                webcamTexture = null;
            }

            // 释放Texture2D资源
            if (texture != null)
            {
                Destroy(texture);
                texture = null;
            }

            // 等待发送线程退出（设置超时）
            if (sendThread != null && sendThread.IsAlive)
            {
                Debug.Log("等待发送线程退出...");
                if (!sendThread.Join(2000))
                {
                    Debug.LogWarning("发送线程未在2秒内退出，将被强制终止");
                    try { sendThread.Abort(); } catch { }
                }
                sendThread = null;
            }

            // 新增：等待接收线程退出
            if (receiveThread != null && receiveThread.IsAlive)
            {
                Debug.Log("等待接收线程退出...");
                if (!receiveThread.Join(2000))
                {
                    Debug.LogWarning("接收线程未在2秒内退出，将被强制终止");
                    try { receiveThread.Abort(); } catch { }
                }
                receiveThread = null;
            }

            Debug.Log("断开连接完成");
        }
        catch (Exception e)
        {
            Debug.LogError("断开连接时出错: " + e.Message);
        }
    }

    /// <summary>
    /// 滑动窗口多数表决，过滤离散手势结果
    /// </summary>
    /// <param name="history">手势历史队列</param>
    /// <param name="newGesture">新识别的手势</param>
    /// <param name="windowSize">窗口大小</param>
    /// <returns>过滤后的手势结果</returns>
    private string FilterGesture(Queue<string> history, string newGesture, int windowSize)
    {
        // 1. 将新结果加入历史队列
        history.Enqueue(newGesture);

        // 2. 保证队列长度不超过窗口大小（移除最旧的结果）
        while (history.Count > windowSize)
        {
            history.Dequeue();
        }

        // 3. 统计队列中每个手势的出现次数
        Dictionary<string, int> gestureCount = new Dictionary<string, int>();
        foreach (string gesture in history)
        {
            if (gestureCount.ContainsKey(gesture))
                gestureCount[gesture]++;
            else
                gestureCount[gesture] = 1;
        }

        // 4. 找到出现次数最多的手势（多数表决）
        int maxCount = 0;
        string majorityGesture = newGesture; // 默认用新结果
        foreach (var pair in gestureCount)
        {
            // 优先选择次数更多的
            if (pair.Value > maxCount)
            {
                maxCount = pair.Value;
                majorityGesture = pair.Key;
            }
            // 若次数相同，优先保留最近出现的（取队列中最后一次出现的）
            else if (pair.Value == maxCount)
            {
                foreach (string g in history.Reverse()) // 从后往前找
                {
                    if (gestureCount[g] == maxCount)
                    {
                        majorityGesture = g;
                        break;
                    }
                }
            }
        }

        return majorityGesture;
    }


    void ProcessGestureData(GestureData gestureData)
    {
        bool is_left_detected = false;
        bool is_right_detected = false;
        // 遍历每只手
        foreach (HandData hand in gestureData.hands)
        {
            if (hand.hand_type == "Left")
            {
                is_left_detected = true;

                // 手势滑动窗口过滤
                filteredLeftGesture = FilterGesture(leftGestureHistory, hand.hand_gesture, gestureWindowSize);
                Debug.Log("Left filtered: " + filteredLeftGesture);
                left_result.text = filteredLeftGesture;

                // 卡尔曼滤波
                landmarkFilter.FilterLeftHandLandmarks(hand.landmarks);

                foreach (HandLandmarkData landmark in hand.landmarks)
                {
                    left_spheres[landmark.id].transform.localPosition = new Vector3(landmark.x, landmark.y, landmark.z * 4);
                }

                // 0
                Vector3 wristPosition = new Vector3(hand.landmarks[0].x, hand.landmarks[0].y, hand.landmarks[0].z * 4);
                Vector3 indexPosition = new Vector3(hand.landmarks[5].x, hand.landmarks[5].y, hand.landmarks[5].z * 4);
                Vector3 middlePosition = new Vector3(hand.landmarks[9].x, hand.landmarks[9].y, hand.landmarks[9].z * 4);
                Vector3 vectorToMiddle = middlePosition - wristPosition;
                Vector3 vectorToIndex = indexPosition - wristPosition;

                Vector3.OrthoNormalize(ref vectorToMiddle, ref vectorToIndex);
                Vector3 normalVector = Vector3.Cross(vectorToIndex, vectorToMiddle);
                left_points[0].transform.localRotation = Quaternion.LookRotation(normalVector, vectorToIndex) * Quaternion.Euler(0, 90, 90);

                left_points[0].transform.localRotation = Quaternion.Euler(
                    left_points[0].transform.localRotation.eulerAngles.x,
                    -left_points[0].transform.localRotation.eulerAngles.y,
                    -left_points[0].transform.localRotation.eulerAngles.z
                );

                HandLandmarkData mark0 = hand.landmarks[0];
                left_points[0].transform.localPosition = new Vector3(mark0.x / 2 - 0.15f, -mark0.y / 2 - 0.3f, mark0.z + hand.bound_area);

                // 1-5
                for (int i = 0; i < 5; i++)
                {
                    for (int j = 1; j <= 4; j++)
                    {
                        if (j < 4)
                        {
                            int index = i * 4 + j;
                            HandLandmarkData nextMark = hand.landmarks[index];
                            Vector3 direction = left_spheres[index + 1].transform.position - left_spheres[index].transform.position;
                            left_points[index].transform.forward = direction.normalized;

                            // 保持当前的旋转，只修改Z轴
                            Quaternion currentRotation = left_points[index].transform.localRotation;

                            // 提取当前的X和Y旋转，固定Z轴
                            Vector3 euler = currentRotation.eulerAngles;
                            euler.z = origin_leftRotations[index].z;

                            // 应用新的旋转
                            left_points[index].transform.localRotation = Quaternion.Euler(euler);
                        }
                    }
                }
            }
            else if (hand.hand_type == "Right")
            {
                is_right_detected = true;

                // 2. 手势滑动窗口过滤
                filteredRightGesture = FilterGesture(rightGestureHistory, hand.hand_gesture, gestureWindowSize);
                Debug.Log("Right filtered: " + filteredRightGesture);
                right_result.text = filteredRightGesture;

                // 卡尔曼滤波
                landmarkFilter.FilterRightHandLandmarks(hand.landmarks);

                foreach (HandLandmarkData landmark in hand.landmarks)
                {
                    right_spheres[landmark.id].transform.localPosition = new Vector3(landmark.x, landmark.y, landmark.z * 4);
                }


                // 0
                Vector3 wristPosition = new Vector3(hand.landmarks[0].x, hand.landmarks[0].y, hand.landmarks[0].z * 4);
                Vector3 indexPosition = new Vector3(hand.landmarks[5].x, hand.landmarks[5].y, hand.landmarks[5].z * 4);
                Vector3 middlePosition = new Vector3(hand.landmarks[9].x, hand.landmarks[9].y, hand.landmarks[9].z * 4);
                Vector3 vectorToMiddle = middlePosition - wristPosition;
                Vector3 vectorToIndex = indexPosition - wristPosition;

                Vector3.OrthoNormalize(ref vectorToMiddle, ref vectorToIndex);
                Vector3 normalVector = Vector3.Cross(vectorToIndex, vectorToMiddle);
                right_points[0].transform.localRotation = Quaternion.LookRotation(normalVector, vectorToIndex) * Quaternion.Euler(0, 90, -90);
                right_points[0].transform.localRotation = Quaternion.Euler(
                    right_points[0].transform.localRotation.eulerAngles.x,
                    -right_points[0].transform.localRotation.eulerAngles.y,
                    -right_points[0].transform.localRotation.eulerAngles.z
                );


                HandLandmarkData mark0 = hand.landmarks[0];
                right_points[0].transform.localPosition = new Vector3(mark0.x / 2 + 0.15f, -mark0.y / 2 - 0.3f, mark0.z + hand.bound_area);

                // 1-5
                for (int i = 0; i < 5; i++)
                {
                    for (int j = 1; j <= 4; j++)
                    {
                        if (j < 4)
                        {
                            int index = i * 4 + j;
                            HandLandmarkData nextMark = hand.landmarks[index];
                            Vector3 direction = right_spheres[index + 1].transform.position - right_spheres[index].transform.position;
                            right_points[index].transform.forward = direction.normalized;

                            // 保持当前的旋转，只修改Z轴
                            Quaternion currentRotation = right_points[index].transform.localRotation;

                            // 提取当前的X和Y旋转，固定Z轴
                            Vector3 euler = currentRotation.eulerAngles;
                            euler.z = origin_rightRotations[index].z;

                            // 应用新的旋转
                            right_points[index].transform.localRotation = Quaternion.Euler(euler);
                        }
                    }
                }
            }
        }
        left_hand.SetActive(is_left_detected);
        right_hand.SetActive(is_right_detected);

        // 手势交互
        left_hold.hold = right_hold.hold = false;
        if (filteredLeftGesture.Equals("7") && is_left_detected)
        {
            left_hold.hold = true;
        }
        if (filteredRightGesture.Equals("7") && is_right_detected)
        {
            right_hold.hold = true;
        }
        if(left_hold.hold && right_hold.hold)
        {
            left_hold.hold = right_hold.hold = false;

            float currentHandDistance = Vector3.Distance(left_hold.transform.position, right_hold.transform.position);
            float scaleFactor = currentHandDistance / lastHandDistance;
            target.transform.localScale *= scaleFactor;
        }
        LastLeftHandPosition = left_hold.transform.position;
        LastRightHandPosition = right_hold.transform.position;
        lastHandDistance = Vector3.Distance(LastLeftHandPosition, LastRightHandPosition);
    }
}