## 1. 项目简介

本项目构建了一个基于 BERT + PyTorch 的金融文本情感分类模型，并通过 Flask API 提供推理服务。模型及服务进一步被封装为 Docker 镜像，最终部署到 Kubernetes 集群，并支持：

- 自动扩缩容（HPA）

- CI/CD 自动部署（Jenkins + Harbor）

- 可观测性监控（Prometheus + Grafana）

该项目模拟真实企业中的 AI 推理服务全链路工程化流程：
数据 → 模型训练 → 推理 API → Docker 化 → K8s 部署 → 自动伸缩 → CI/CD → 监控


## 2. 项目架构

<br>
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/dfafd0ee-0200-4f62-ac07-62245c874823" />
<br> 


## 3. 文件结构

```bash
FINANCIAL_NLP_PROJECT/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── Sentences_AllAgree.txt
├── model/
│   ├── model.pth                # BERT 权重
│   ├── label_encoder.pkl        # 标签编码器
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
├── app.py                       # Flask 推理 API
├── train.py                     # 模型训练脚本
├── data_preparation.py          # 数据清洗脚本
├── Download_model.py            # 下载预训练模型（可选）
├── Dockerfile                   # Docker 镜像构建文件
├── Jenkinsfile                  # CI/CD 流水线
└── requirements.txt
```


## 4. 模型训练

### 4.1 数据清洗

数据来自 HuggingFace 的 Financial PhraseBank，其中的 Sentences_AllAgree 文件包含了标注一致性最高的金融文本样本。每条数据的格式如下：

```bash
"Construction volumes meanwhile grow at a rate of 10-15 percent annually .@positive"
```
文本内容紧接一个情感标签（@positive / @negative / @neutral）。
<br> 
<img width="865" height="734" alt="image" src="https://github.com/user-attachments/assets/b0e89825-80a3-41d6-8918-617ebdffd18d" />
<br> 
数据准备脚本（data_preparation.py）完成了以下任务：

- 逐行解析原始 TXT 文件

- 移除标签标记（@positive / @negative / @neutral）

- 生成标准化结构：text、label 两列

- 转换为模型可直接使用的DataFrame 结构化数据

- 按 80/20 进行分层切分成80% → 训练集（train.csv）、20% → 验证集（val.csv）

最终生成：

```bash
data/train.csv
data/val.csv
```

<br>
<img width="865" height="205" alt="image" src="https://github.com/user-attachments/assets/c20607f7-79c2-469b-84b3-957d3b33aada" />
<br> 




### 4.2 模型训练

项目使用 BERT-base-uncased 作为文本编码模型，并在其基础上微调金融文本情感分类任务（positive / negative / neutral）。训练脚本位于 train.py。

#### 4.2.1 模型数据加载

训练脚本使用 FinancialDataset 类对文本进行编码：

```bash
encodings = tokenizer(text,
                      truncation=True,
                      padding="max_length",
                      max_length=128,
                      return_tensors="pt")
```

该步骤完成：

- tokenization：将文本拆成 BERT 的 token

- 转换 token IDs：映射为数字序列

- 生成 attention_mask：标记有效 token

- 固定长度（padding/truncation）：保证 batch 尺寸一致

随后由 DataLoader 生成 batch 数据用于训练。

#### 4.2.2 模型定义

使用 HuggingFace Transformers 提供的预训练模型：

```bash
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)
```

此处完成：

- 加载预训练 BERT 模型（encoder 部分参数已训练）

- 创建随机初始化的分类头（Linear 层）

- 设置输出类别数量 num_labels=3（即nn.Linear(768, 3)）

设置优化器 optimizer 与学习率调度器 scheduler。

#### 4.2.3 训练循环

训练过程包括前向传播、反向传播和参数优化：

```bash
outputs = model(**inputs)
loss = outputs.loss
loss.backward()

optimizer.step()
scheduler.step()
optimizer.zero_grad()
```
同时每个 epoch 结束后会在验证集上计算准确率：

```bash
preds = torch.argmax(outputs.logits, dim=1)
acc = (preds == labels).float().mean()
```

确保模型在训练过程中不断评估性能，避免过拟合。

训练过程中会输出：

- 每个 batch 的 loss

- 每个 epoch 的平均 loss

- 验证集准确率

<br>
<img width="865" height="303" alt="image" src="https://github.com/user-attachments/assets/b518e1b7-da62-4f75-8b12-23d6592abad0" />
<br>

### 4.3 模型输出与测试

训练结束后，模型权重与标签编码器会保存到 model/ 目录

保存代码：

```bash
torch.save(model.state_dict(), "model/model.pth")
joblib.dump(label_encoder, "model/label_encoder.pkl")
```

这些文件将在推理 API (app.py) 中被加载。

模型推理测试：

```bash
predict.py 'Xiaomi car sales reach a new high, with users generally expressing high satisfaction'
```

<br>
<img width="865" height="145" alt="image" src="https://github.com/user-attachments/assets/13a5ac9b-648f-4c20-9ba9-1e7074c7b01d" />
<br>

结果符合预期。
<br>

## 5. 模型推理 API

本项目使用 Flask 封装 BERT 文本分类模型，提供 REST 风格推理接口，实现在线预测能力。

推理代码位于 app.py。

### 5.1 模型加载逻辑

推理服务启动时会加载三个核心组件：

1）Tokenizer（文本编码器）

来自与训练一致的预训练模型：

```bash
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

用于将用户输入的文本转成：

- input_ids

- attention_mask

2）标签编码器（label encoder）

用于将预测的数字类别映射回文字标签（positive / negative / neutral）

3）模型结构 + 训练权重

推理时重新构建与训练阶段一致的 BERT 结构：

```bash
config = BertConfig.from_pretrained(MODEL_NAME, num_labels=3)
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()
```

流程：

- 先构建一个新的 BERT 分类模型结构

- 再加载训练阶段保存的权重（model.pth）

- model.eval() 关闭 dropout，使推理稳定


### 5.2 推理接口

API 格式：

- 方法：POST

- 路径：/predict

- 输入 JSON：

```bash
{
  "text": "Li Auto Automobile sales have broken through a new high..."
}
```

- 返回 JSON：

```bash
{
  "text": "Li Auto Automobile sales have broken through a new high...",
  "prediction": "positive"
}
```
推理逻辑：

```bash
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
label_name = label_encoder.inverse_transform([pred])[0]
```

## 6. Docker 部署

Dockerfile：

```yaml
FROM python:3.14-slim

WORKDIR /app
COPY . .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install flask transformers joblib scikit-learn requests

EXPOSE 5000
CMD ["python", "app.py"]
```

说明：

- 基于官方 Python slim 镜像

- 安装 CPU 版本 PyTorch（轻量且适合 API 推理）

- 安装 Flask、Transformers 等依赖

- 默认启动 Flask API

### 6.1 构建与运行 Docker

1）构建镜像

```bash
docker build -t financial-nlp-api:v1 .
```
<br>
<img width="865" height="312" alt="image" src="https://github.com/user-attachments/assets/48c0cc61-4c48-4c4f-a873-5928d18569d4" />
<br>

构建成功

2）运行容器
```bash
docker run -d -p 5000:5000 financial-nlp-api:v1
```
<br>
<img width="865" height="35" alt="image" src="https://github.com/user-attachments/assets/d647c1d9-5b3c-4ea9-b433-fdc9c742e467" />
<br>
<img width="865" height="24" alt="image" src="https://github.com/user-attachments/assets/0117aba9-4412-469c-8e45-4e806982cc3d" />
<br>

### 6.2 测试容器服务

中性新闻示例

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"This stock looks promising"}'
```

返回：

```bash
{"prediction":"neutral","text":"This stock looks promising"}
```

<br>
<img width="865" height="20" alt="image" src="https://github.com/user-attachments/assets/96bb4ae3-675a-4a57-81d7-03ab63ebaf7d" />

<br>
<img width="865" height="53" alt="image" src="https://github.com/user-attachments/assets/e70649bc-59ee-4e60-87ba-19cd64e55ac1" />
<br>

## 7. K8s 部署

将训练好的金融文本分类 API 以容器形式部署到 Kubernetes 集群，实现服务化运行与集群级管理。

### 7.1 Deployment 配置（deployment.yaml）

Deployment 用于定义：

- 容器镜像

- 副本数量（replicas）

- 容器端口

- 资源限制

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-nlp-deploy
  labels:
    app: financial-nlp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: financial-nlp
  template:
    metadata:
      labels:
        app: financial-nlp
    spec:
      containers:
        - name: financial-nlp
          image: financial-nlp-api:v1
          ports:
            - containerPort: 5000
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "1"
              memory: "2Gi"
```  
说明：

- containerPort: 5000 为 Flask API 的端口

- resources 配置为后续 HPA 提供 CPU 指标依据

### 7.2 Service 配置（service.yaml）

通过 NodePort 方式对外暴露 API

```yaml
apiVersion: v1
kind: Service
metadata:
  name: financial-nlp-svc
spec:
  type: NodePort
  selector:
    app: financial-nlp
  ports:
    - port: 5000
      targetPort: 5000
      nodePort: 30500
```

说明：

- 外部访问端口：30500

- 集群内部服务端口：5000

- 选择器与 Deployment 保持一致，通过标签 app=financial-nlp 匹配 Pod


### 7.3 应用部署配置

应用 Deployment 与 Service：

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

<br>
<img width="865" height="203" alt="image" src="https://github.com/user-attachments/assets/91ac3ec6-9047-4096-a371-1cf44aa409a4" />
<br>
 
初次部署截图显示： ErrImagePull

进入Pod 内部验证

```bash
kubectl exec -it financial-nlp-deploy-xxx -- sh
python app.py
```

初次运行报错（无法访问 huggingface.co）：

```bash
ReadTimeout: thrown while requesting HEAD https://huggingface.co/...
```

<br>
<img width="865" height="239" alt="image" src="https://github.com/user-attachments/assets/8eaa2c8f-3950-4953-aba3-24d42fa2b766" />
<br>

原因：
容器内不能访问外网 → tokenizer 和模型未下载。

解决方案：
提前将模型/ tokenizer 下载到本地并打包进镜像（通过 Download_model.py 解决）。

<br>
<img width="865" height="394" alt="image" src="https://github.com/user-attachments/assets/c06f2038-1d55-4906-8d15-a96804a914eb" />
<br>

### 7.4 使用 NodePort 本地访问服务

查询 Service：
```bash
kubectl get svc financial-nlp-svc
```

输出示例：
```bash
financial-nlp-svc   NodePort   5000:30500/TCP
```

在宿主机上测试：
```bash
curl -X POST http://localhost:30500/predict \
-H "Content-Type: application/json" \
-d '{"text":"Li Auto stock rises after record sales"}'
```

返回：
```bash
{"prediction":"positive","text":"Li Auto stock rises after record sales"}
```

<br>
<img width="865" height="395" alt="image" src="https://github.com/user-attachments/assets/404d2c5f-a646-4827-acff-2c954ae025bb" />
<br>

说明：
- Pod 内模型成功加载

- K8s 部署正常运行

- NodePort 对外访问成功


## 8. 自动扩缩容（HPA）

为文本分类 API 配置 Kubernetes 自动扩缩容能力，使服务能够根据 CPU 压力自动调节 Pod 数量，从而实现弹性伸缩与高可用。

### 8.1 启动与验证Metrics Server

HPA 需要依赖 Metrics Server 提供 CPU/内存指标。

检查状态：

```bash
kubectl get pods -n kube-system | grep metrics
kubectl get apiservices | grep metrics
```

示例输出：

```bash
metrics-server-576c8c997c-61njk     1/1 Running
v1beta1.metrics.k8s.io              True
```
表示 Metrics Server 正常运行


检查节点和Pod CPU/内存：

```bash
kubectl top nodes
kubectl top pods
```

示例：

```bash
financial-nlp-deploy   CPU(cores)=1m   Memory=1406Mi
```
说明 Metrics Server 可以正常获取指标 → HPA 可以使用。

<br>
<img width="865" height="180" alt="image" src="https://github.com/user-attachments/assets/df4a09da-1f9d-4c05-8f22-66e572d142e3" />
<br>

### 8.2 创建 HPA（目标 CPU：60%）

使用自动化命令创建 HPA：

```bash
kubectl autoscale deployment financial-nlp-deploy \
  --cpu-percent=60 \
  --min=1 \
  --max=5
```

<br>
<img width="865" height="95" alt="image" src="https://github.com/user-attachments/assets/f14628a4-c2b1-44a4-a3ae-1a33801ae2db" />
<br>


查看：

```bash
kubectl get hpa
```

初始状态：
```bash
NAME                TARGETS   MINPODS   MAXPODS
financial-nlp-deploy  0%/60%      1         5
```
<br>
<img width="865" height="238" alt="image" src="https://github.com/user-attachments/assets/95c4aa2f-5e39-4f3e-84df-082647323880" />
<br>


### 8.3 执行高并发压测（触发扩缩容）

使用 10 个线程 × 每线程 20 次请求：

```bash
for j in {1..10}; do
(
    for i in {1..20}; do
        curl -s -X POST http://localhost:30500/predict \
        -H "Content-Type: application/json" \
        -d '{"text":"Bank profits drop amid interest rate cuts"}' > /dev/null
    done
) &
done

wait
echo "High Load Test Done"
```

约 200 次推理，可显著提升 CPU 使用率，适合触发 HPA。

实时观察：

```bash
watch -n 2 kubectl get hpa
watch -n 2 kubectl get pods
```
CPU 使用率上升：

```bash
TARGETS: 199%/60%
```
Pod 从 1 个扩容到 4 个：

```bash
financial-nlp-deploy-xxxxx   Running
financial-nlp-deploy-xxxxx   Running
financial-nlp-deploy-xxxxx   Running
financial-nlp-deploy-xxxxx   Running
```

<br>
<img width="865" height="308" alt="image" src="https://github.com/user-attachments/assets/19f4b2c5-82ae-4abb-b5af-b724777e2cf2" />
<br>

### 8.4 容灾验证（删除 Pod → 自动恢复）

删除一个正在运行的 Pod：
```bash
kubectl delete pod financial-nlp-deploy-xxxxx
```
<br>
<img width="865" height="56" alt="image" src="https://github.com/user-attachments/assets/d72e46a8-c91a-4b49-96b1-53f81471b779" />
<br>

观察：

```bash
Terminating...
Running (新 Pod 启动)
```

说明 ReplicaSet 自动拉起新的 Pod，验证了 K8s 自愈能力。

<br>
<img width="865" height="169" alt="image" src="https://github.com/user-attachments/assets/ed8532e4-25fb-40b1-a580-327a9a90d507" />
<br>
<img width="865" height="211" alt="image" src="https://github.com/user-attachments/assets/ea224731-f0ab-4bc5-a76c-09d0ed0ef36e" />
<br>


## 9. CI/CD（Jenkins + Harbor）

通过 Jenkins 流水线实现完整的 CI/CD 流程：

- 自动构建镜像（Build）

- 推送到 Harbor 私有仓库（Push）

- 自动触发 Kubernetes 滚动更新（Deploy）

该流程实现了模型迭代 → 自动部署上线的完整工程能力。

本项目沿用上个项目的 Jenkins/Harbor 环境。

### 9.1 Jenkinsfile（完整流水线脚本）

```yaml
pipeline {
    agent any

    environment {
        HARBOR_URL = "192.168.0.137:8081"
        HARBOR_PROJECT = "financial-nlp"
        IMAGE_NAME = "financial-nlp-api"
        IMAGE_TAG = "v1"
    }

    stages {

        stage('Build Image') {
            steps {
                sh """
                    docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG} .
                """
            }
        }

        stage('Push Image to Harbor') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'harbor-cred', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
                    sh """
                        echo $PASS | docker login ${HARBOR_URL} -u $USER --password-stdin
                        docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}
                    """
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([file(credentialsId: 'k8s-config', variable: 'KUBECONFIG')]) {
                    sh """
                        kubectl --kubeconfig=$KUBECONFIG rollout restart deployment/financial-nlp-deploy
                    """
                }
            }
        }
    }
}
```

<br>
<img width="865" height="361" alt="image" src="https://github.com/user-attachments/assets/0be46f2e-3446-43e9-96d3-e8c8478f90fb" />
<img width="865" height="387" alt="image" src="https://github.com/user-attachments/assets/349b5f28-5385-4b81-93d1-f3069e849124" />
<br>

### 9.2 pipeline阶段解析
（1）Build Image — 构建 Docker 镜像

```bash
docker build -t 192.168.0.137:8081/financial-nlp/financial-nlp-api:v1 .
```

功能：

- 从 Dockerfile 构建模型推理服务镜像

- 注入 Flask、模型权重、tokenizer、依赖包

- 镜像 TAG 由 Jenkins 管理（v1、v2… 可迭代）

（2）Push Image to Harbor — 推送镜像

使用 Harbor 凭证（harbor-cred）：

```bash
USER / PASS
```

登录 Harbor 并推送：

```bash
docker push 192.168.0.137:8081/financial-nlp/financial-nlp-api:v1
```

Harbor UI 中可看到上传成功的镜像：

<br>
<img width="865" height="407" alt="image" src="https://github.com/user-attachments/assets/9dd86f63-276a-4be1-842f-b29a9b8d6cff" />
<br>

（3）Deploy to Kubernetes — 自动滚动更新

Jenkins 通过 k8s 凭证（k8s-config）访问集群：

```bash
kubectl --kubeconfig=$KUBECONFIG rollout restart deployment/financial-nlp-deploy
```

作用：

- 触发 Deployment 进行滚动更新

- 旧 Pod 逐个 Terminating

- 新版本镜像的 Pod 自动创建

<br>
<img width="865" height="157" alt="image" src="https://github.com/user-attachments/assets/e8262839-51e6-4b3b-9f4d-b4c786de4c47" />
<br>
<img width="865" height="151" alt="image" src="https://github.com/user-attachments/assets/444ff95b-a9d6-48ec-8118-bd2ac885f4b2" />
<br>

截图显示：

```bash
Pulled image "192.168.0.137:8081/financial-nlp/financial-nlp-api:v1"
Started container financial-nlp
```
### 9.3 服务验证

```bash
curl -X POST http://localhost:30500/predict \
-H "Content-Type: application/json" \
-d '{"text":"Bank profits drop amid interest rate cuts"}'
```

返回：

```bash
{"prediction":"negative","text":"Bank profits drop amid interest rate cuts"}
```

<br>
<img width="865" height="46" alt="image" src="https://github.com/user-attachments/assets/bbcd193d-9a4b-417d-9e9f-145680a401c9" />
<br>

说明：

- 镜像已更新

- 新版本 API 成功运行

- NodePort 访问正常


## 10. 模型服务监控（Prometheus + Grafana）

本节为部署在 Kubernetes 中的 金融文本分类服务 配置系统级监控，包括：

- Pod CPU 使用率

- Pod 内存使用率

- 实时趋势图

多副本（HPA 扩容后）对比展示

监控组件沿用上个项目 Helm 安装的 kube-prometheus-stack（含 Prometheus、Grafana、Alertmanager、NodeExporter）。

### 10.1 检查监控组件

列出现有监控 Pod：

```bash
kubectl get pods -n monitor
```

示例输出（与你截图一致）：

```bash
monitor-grafana-xxxxx               3/3   Running
monitor-kube-prometheus-st-operator-xxxxx  1/1 Running
monitor-prometheus-node-exporter-xxxxx    1/1 Running
prometheus-monitor-kube-prometheus-st-prometheus-0   2/2 Running
```

<br>
<img width="865" height="145" alt="image" src="https://github.com/user-attachments/assets/b2cc09f5-2d1e-4eb7-8e2d-2df7b2f18e27" />
<br>

说明监控系统已正常启动。

### 10.2 访问 Grafana

Grafana 默认在 Kubernetes 内部运行，需要通过端口转发访问：

```bash
kubectl port-forward -n monitor svc/monitor-grafana 3000:80
```

访问地址：

```bash
http://localhost:3000
```
默认登录：

- 用户名：admin
- 密码通过以下命令获取：

```bash
kubectl get secret -n monitor monitor-grafana -o jsonpath="{.data.admin-password}" | base64 --decode; echo
```

<br>
<img width="865" height="99" alt="image" src="https://github.com/user-attachments/assets/dda7eeaf-21c0-4a74-8692-ff8d58b4d7cf" />
<br>

访问 Prometheus（可选）

```bash
kubectl port-forward -n monitor svc/monitor-kube-prometheus-st-prometheus 9090:9090
```

<br>
<img width="865" height="65" alt="image" src="https://github.com/user-attachments/assets/a076d26a-f0bb-4a35-b992-dfb08980bacf" />
<br>


### 10.3 Grafana 新建 Dashboard

在 Grafana 中：

Dashboard → New Dashboard → Add Panel → 选择 Prometheus 数据源

#### 10.3.1 CPU 使用率（毫核 / mCPU）

使用服务 Pod 标签：

```bash
app=financial-nlp
```
PromQL：

```bash
kubectl port-forward -n monitor svc/monitor-kube-prometheus-st-prometheus 9090:9090
```
说明：

- container_cpu_usage_seconds_total 为 CPU 时间

- rate() 取 2 分钟窗口的变化速率

- *1000 转成 mCPU（毫核）

- 每个 Pod 单独显示趋势

<br>
<img width="865" height="701" alt="image" src="https://github.com/user-attachments/assets/7a386f9b-7f15-42d4-a9ca-875bb9e71890" />
<br>

#### 10.3.2 内存使用（MiB）

PromQL：

```bash
sum(container_memory_usage_bytes{pod=~"financial-nlp-deploy.*", container!="POD"}) 
by (pod) / 1024 / 1024
```
说明：

- container_memory_usage_bytes 为容器内存占用

- 除以 1024² 转成 MiB

- HPA 扩容时，可以观察多个 Pod 的内存使用对比


<br>
  <img width="865" height="678" alt="image" src="https://github.com/user-attachments/assets/3adfd1dc-fcc8-4362-a851-6e537d853b41" />
<br>
