# ai-detector — AI 视频检测分析平台

独立的 AI 分析平台，从标准 RTSP 流中拉取视频帧，执行 YOLO 目标检测、入侵检测、越线检测，并提供报警推送和管理 Web UI。

与 `miloco-camera`（流媒体网关）解耦，通过标准 RTSP 协议对接任意品牌摄像头。

---

## 架构说明

```
miloco-camera（流媒体网关）         ai-detector（AI 分析平台）
  小米摄像头 ─→ XiaomiAdapter            RTSP 拉流
  海康摄像头 ─→ RtspAdapter     ──RTSP──→  PyAV 解码
  大华摄像头 ─→ RtspAdapter              YOLO / OpenCV 检测
                 ↓                        报警落库 + 截图
            MediaMTX RTSP 中继            Webhook / 邮件推送
```

---

## 快速启动

### 方式一：Docker Compose

```yaml
# docker-compose.yaml
services:
  ai-detector:
    build: .
    container_name: ai-detector
    restart: unless-stopped
    ports:
      - "8090:8090"
    volumes:
      - ./data:/app/data
    environment:
      - GATEWAY_URL=http://miloco-camera:8080   # miloco-camera 地址
      - YOLO_MODEL=yolo11n.pt
      - ALERT_COOLDOWN=10
```

```bash
docker compose up -d --build
```

访问：http://localhost:8090

### 方式二：与 miloco-camera 协同部署

在 miloco-camera 同一 `docker-compose.yaml` 中追加：

```yaml
  ai-detector:
    build: ../ai-detector
    container_name: ai-detector
    restart: unless-stopped
    ports:
      - "8090:8090"
    volumes:
      - ./ai-data:/app/data
    environment:
      - GATEWAY_URL=http://miloco-camera:8080
```

---

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GATEWAY_URL` | `http://localhost:8080` | miloco-camera 网关地址，用于自动同步流源 |
| `YOLO_MODEL` | `yolo11n.pt` | YOLO 默认模型文件名 |
| `YOLO_CONFIDENCE` | `0.5` | 默认置信度阈值 |
| `ALERT_COOLDOWN` | `10` | 同一路流同类型报警冷却时间（秒） |
| `SMTP_ENABLED` | `false` | 是否启用邮件推送 |
| `SMTP_HOST` | `smtp.qq.com` | SMTP 服务器 |
| `SMTP_PORT` | `465` | SMTP 端口 |
| `SMTP_USER` | - | 发件邮箱 |
| `SMTP_PASSWORD` | - | 邮箱授权码 |
| `SMTP_TO` | - | 收件邮箱（多个用逗号分隔） |
| `WEBHOOK_URL` | - | Webhook 推送地址（企业微信/飞书/钉钉机器人） |
| `PORT` | `8090` | 服务端口 |

---

## 使用流程

### 1. 添加流源

进入 **流源配置** 页面，点击"添加流源"：

```
名称：前门摄像头
RTSP 地址：rtsp://192.168.1.1:8554/camera1
```

或点击"**从网关同步**"，自动从 miloco-camera 导入所有 RTSP 流。

### 2. 配置检测算法

进入 **算法配置** 页面，为每路流添加检测器：

**YOLO 目标检测**
```json
{
  "model": "yolo11n.pt",
  "confidence": 0.5,
  "classes": ["person", "car"],
  "iou": 0.45,
  "detect_interval": 1.0
}
```

**入侵区域检测（ROI）**
```json
{
  "roi": [[100,100],[500,100],[500,400],[100,400]],
  "min_area": 500,
  "sensitivity": 50,
  "detect_interval": 1.0
}
```

**越线检测（虚拟绊线）**
```json
{
  "lines": [[[0,300],[1280,300]]],
  "direction": "any",
  "min_area": 300,
  "detect_interval": 1.0
}
```

### 3. 查看报警

进入 **报警查询** 页面，可按流源、类型、时间范围筛选，查看截图。

### 4. 系统监控

进入 **系统监控** 页面，查看 CPU/内存/磁盘/GPU 资源占用，以及各路流的帧率和检测任务状态。

---

## API 文档

启动后访问：`http://localhost:8090/docs`

主要接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/streams` | 流源列表 |
| `POST` | `/api/streams` | 添加流源 |
| `POST` | `/api/streams/sync/gateway` | 从网关同步流源 |
| `GET` | `/api/streams/{id}/test` | 测试 RTSP 连通性 |
| `GET` | `/api/detections` | 检测配置列表 |
| `POST` | `/api/detections` | 添加检测配置 |
| `POST` | `/api/detections/{id}/toggle` | 启用/停用检测 |
| `GET` | `/api/alerts` | 报警记录查询（分页+筛选） |
| `GET` | `/api/alerts/stats` | 报警统计 |
| `GET` | `/api/system/stats` | 系统资源 |
| `GET` | `/api/system/processes` | 流和检测任务状态 |

---

## GPU 加速

如需 NVIDIA GPU 解码和推理，在 docker-compose.yaml 中取消注释：

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

并在检测配置的 JSON 中指定 `"device": "cuda"`：

```json
{"model": "yolo11n.pt", "device": "cuda", "confidence": 0.5}
```
