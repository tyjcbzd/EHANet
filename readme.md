# :sunflower:: LANet: Lightweight Attention Network for Medical Image Segmentation 
**This is the official implementation for article "Edge-guided and Hierarchical Aggregation Network for Robust Medical Image Segmentation".** 



## Overview

### Module 1
**The EFA block enhances the model's feature extraction capability by capturing task-relevant information while reducing redundancy in channel and spatial locations.**



### Module 2
**The AFF decoding block fuses the purified low-level features from the encoder with the sampled features from the decoder, enhancing the network's understanding and expression of input features.**


## 	:pencil: Requirements
* torch == 2.1.1+cu121
* tensorboard == 2.11.2
* numpy == 1.24.1
* python == 3.9.18
* torchvision == 0.16.1+cu121
* ...

## 	 :bar_chart: Datasets
The efficiency of LANet was evaluated using four public datasets: kvasir-SEG, CVC-clinicDB, CVC-colonDB, and the Data Science Bowl 2018. 
All datasets used in paper are public, you can download online.

Split the datasets for train, validation and test with ratio **8:1:1**

##   :chart_with_upwards_trend: Results
### Quantitative results


### Qualitative results

### Ablation study

## :black_nib: For citation
waiting...

:exclamation: :eyes: **The codes can not be used for commercial purposes!!!**