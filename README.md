## 1.项目简介

本项目构建了一个基于 BERT + PyTorch 的金融文本情感分类模型，并通过 Flask API 提供推理服务。模型及服务进一步被封装为 Docker 镜像，最终部署到 Kubernetes 集群，并支持：

自动扩缩容（HPA）

CI/CD 自动部署（Jenkins + Harbor）

可观测性监控（Prometheus + Grafana）

该项目模拟真实企业中的 AI 推理服务全链路工程化流程：
数据 → 模型训练 → 推理 API → Docker 化 → K8s 部署 → 自动伸缩 → CI/CD → 监控


## 2.项目架构

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/dfafd0ee-0200-4f62-ac07-62245c874823" />
<br> 


## 3.文件结构
<br> 

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


## 4.模型训练

### 4.1 数据清洗

### 4.2 模型训练

### 4.3 模型推理




