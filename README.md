# Medical Image Classification using AWS SageMaker

This repository demonstrates an end-to-end Machine Learning pipeline developed on AWS SageMaker for medical image classification. The project covers data ingestion, hyperparameter tuning, model training with debugging/profiling, and deployment to a real-time endpoint.

## 📁 Project Structure

* **`Untitled.ipynb`**: Main Jupyter Notebook containing data orchestration, training jobs, and deployment.
* **`hpo.py`**: Training script optimized for SageMaker (Hyperparameter Optimization).
* **`Model_Performance_Report.pdf`**: Comprehensive report detailing model architecture and tuning results.
* **`performance_metrics.png`**: Visualization of the hyperparameter tuning performance and loss reduction.

## 🚀 Overview
The project uses **Transfer Learning** with a pre-trained **ResNet18** model to classify medical images (Pneumonia detection) from X-ray data.

### ⚙️ Hyperparameter Tuning (HPO)
I utilized SageMaker's Hyperparameter Tuner to find the optimal settings:
- **Search Space**: 
  - Learning Rate: Logarithmic range [0.001, 0.1]
  - Batch Size: [2, 4, 8]
- **Best Result**: The most accurate model was achieved with a **Learning Rate of 0.001** and a **Batch Size of 2**, reaching a Final Objective Value (Loss) of **0.3693**.

![Performance Metrics](performance_metrics.png)

## 🛠️ Debugging & Profiling
To ensure efficient training, I integrated **SageMaker Debugger and Profiler**:
* **Monitoring**: Used `SMDebug` to track the loss curve in real-time.
* **Profiling**: Monitored system utilization (CPU/GPU) to identify and resolve performance bottlenecks.

## 🌐 Deployment
The best-performing model was deployed to a real-time inference endpoint on an **`ml.m5.large`** instance, allowing for immediate predictions on new medical imaging data.

---
*Developed by Ghaida Alharbi*
