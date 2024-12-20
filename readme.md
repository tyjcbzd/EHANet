# :rainbow:: EHANet: Edge-guided and Hierarchical Aggregation Network for Robust Medical Image Segmentation

## States
- 15.11 Accepted and release code
- 22.10 Submit revised version
- 29.08 Major revision
- 15.07 Underview
- 10.07.2024 - The article was submitted to Journal "Biomedical Signal Processing and Control" for review. 


## Overview
This paper introduces a novel Edge-guided and Hierarchical Aggregation Network (EHANet) which excels at capturing rich contextual information and preserving fine spatial details, addressing the critical issues of inaccurate mask edges and detail loss prevalent in current segmentation models. The Inter-layer Edge-aware Module (IEM) enhances edge prediction accuracy by fusing early encoder layers, ensuring precise edge delineation. The Efficient Fusion Attention Module (EFA) adaptively emphasizes critical spatial and channel features while filtering out redundancies, enhancing the model's perception and representation capabilities. The Adaptive Hierarchical Feature Aggregation Module (AHFA) module optimizes feature fusion within the decoder, maintaining essential information and improving reconstruction fidelity through hierarchical processing. 


![Image 1](images/Overview.png)

### AHFA
![Image 2](images/AHFA.png)


## Results on Kvasir-SEG
![Image 3](images/5_Kvasir_compare.png)

## Results on DSB 2018
![Image 4](images/6_DSB_compare.png)

## Results on ISIC 2018
![Image 5](images/7_ISIC_compare.png)

## Results on LGG
![Image 6](images/8_LGG_compare.png)

## Grad-cam result
<img src="images/9_grad_cam_compare.png" alt="Image 7" width="300" height="400"/>

## :black_nib: For citation
```bibtex
@article{tang2025edge,
  title={Edge-guided and hierarchical aggregation network for robust medical image segmentation},
  author={Tang, Yi and Zhao, Di and Pertsau, Dmitry and Gourinovitch, Alevtina and Kupryianava, Dziana},
  journal={Biomedical Signal Processing and Control},
  volume={101},
  pages={107202},
  year={2025},
  publisher={Elsevier}
}
```

:exclamation: :eyes: **The codes can not be used for commercial purposes!!!**
