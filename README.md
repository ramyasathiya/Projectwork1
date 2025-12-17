## Title of the Project

Brain Tumor Detection from MRI Images using GAN-based Anomaly Detection
Brain Tumor Detection from MRI Images using GAN-based Anomaly Detection is a deep learning–based medical imaging system that leverages Generative Adversarial Networks (GANs) to identify abnormal tumor regions in brain MRI scans by learning normal brain patterns and detecting deviations, enabling accurate tumor detection even with limited labeled data while supporting faster and more reliable clinical diagnosis.

## Small Description

The project focuses on detecting brain tumors from MRI images using a Generative Adversarial Network (GAN)-based anomaly detection approach, aimed at assisting early diagnosis and improving medical decision-making accuracy.

## About

Brain Tumor Detection from MRI Images using GAN-based Anomaly Detection is a deep learning–driven medical imaging project designed to assist in the early and accurate identification of brain tumors from MRI scans. Conventional tumor detection techniques often rely on fully supervised models that require large volumes of labeled tumor data and extensive manual analysis by radiologists, making the process time-consuming and resource-intensive.

This project introduces a GAN-based anomaly detection approach, where the model is primarily trained on normal (healthy) brain MRI images to learn standard anatomical patterns. Any deviation from this learned distribution is treated as an anomaly, enabling effective detection of tumor regions even with limited labeled abnormal data. The system highlights suspicious areas through reconstruction errors, improving interpretability and diagnostic support.

The proposed solution aims to reduce manual effort, enhance detection accuracy, and support clinicians in making faster and more reliable medical decisions, contributing to the advancement of intelligent and scalable healthcare technologies.

## Features

GAN-based anomaly detection for MRI images

Deep learning–driven tumor localization

Works effectively with limited labeled datasets

High detection accuracy

Reduced manual intervention

Scalable and adaptable to different MRI datasets

## Requirements

Operating System
64-bit OS (Windows 10 / Ubuntu)
Development Environment
Python 3.7 or later
Deep Learning Frameworks
TensorFlow / PyTorch for GAN model training
Image Processing Libraries
OpenCV
NumPy
Pillow
Machine Learning Libraries
scikit-learn
IDE
VS Code / Jupyter Notebook / Google colab
Additional Dependencies
Matplotlib
Pandas
TensorFlow GPU (optional for faster training)

## System Architecture

<img width="829" height="463" alt="image" src="https://github.com/user-attachments/assets/28f850ac-aab6-4f84-872a-a5c509e9f47c" />


## Output
## Output 1 – GAN LOSS CURVES

## Output 2 – ANAMOLY SCORE DISTRIBUTION

## Output 3 - ROC Curve


Detection Accuracy: 96.5%
Note: Accuracy may vary based on dataset and training parameters.

## Results and Impact

The GAN-based brain tumor detection system demonstrates high efficiency in identifying abnormal regions in MRI images without requiring extensive labeled tumor datasets. This approach significantly enhances early tumor detection and reduces diagnostic workload for radiologists.

The project highlights the potential of anomaly detection techniques in medical imaging and contributes to the development of intelligent healthcare systems, enabling faster, more reliable, and cost-effective diagnostic solutions.

## Articles Published / References

Goodfellow, I. et al., “Generative Adversarial Networks,” Advances in Neural Information Processing Systems, 2014.

Schlegl, T. et al., “Unsupervised Anomaly Detection with GANs for Medical Imaging,” Information Processing in Medical Imaging, 2017.

Litjens, G. et al., “A Survey on Deep Learning in Medical Image Analysis,” Medical Image Analysis, 2017.
