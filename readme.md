# :rainbow:: EHANet: Edge-guided and Hierarchical Aggregation Network for Robust Medical Image Segmentation(1)
**This is the official implementation for article "Edge-guided and Hierarchical Aggregation Network for Robust Medical Image Segmentation".** 

## States
- 10.07.2024 - The article is submitted to Journal "Biomedical Signal Processing and Control" for review


## Overview
Medical image segmentation with the convolutional neural networks (CNNs), significantly enhances clinical analysis and disease diagnosis. However, medical images inherently exhibit large intra-class variability, minimal inter-class differences, and substantial noise levels. Extracting robust contextual information and aggregating discriminative features for fine-grained segmentation remains a formidable task. Additionally, existing methods often struggle with producing accurate mask edges, leading to blurred boundaries and reduced segmentation precision. This paper introduces a novel Edge-guided and Hierarchical Aggregation Network (EHANet) which excels at capturing rich contextual information and preserving fine spatial details, addressing the critical issues of inaccurate mask edges and detail loss prevalent in current segmentation models. The Inter-layer Edge-aware Module (IEM) enhances edge prediction accuracy by fusing early encoder layers, ensuring precise edge delineation. The Efficient Fusion Attention Module (EFA) adaptively emphasizes critical spatial and channel features while filtering out redundancies, enhancing the model's perception and representation capabilities. The Adaptive Hierarchical Feature Aggregation Module (AHFA) module optimizes feature fusion within the decoder, maintaining essential information and improving reconstruction fidelity through hierarchical processing. Quantitative and qualitative experiments on four public datasets demonstrate the effectiveness of EHANet in achieving superior mIoU, mDice, and edge accuracy against six other state-of-the-art segmentation methods, highlighting its robustness and precision in diverse clinical scenarios.



## 	:pencil: Requirements
* torch == 2.1.1+cu121
* tensorboard == 2.11.2
* numpy == 1.24.1
* python == 3.9.18
* torchvision == 0.16.1+cu121
* ...


## :black_nib: For citation
waiting...

:exclamation: :eyes: **The codes can not be used for commercial purposes!!!**
