# GBA-Net
This repository is the work of "GBA-Net: A Method for Brain Tumor Segmentation Based on Multi-scale Gaussian Boundary Attention" based on pytorch implementation. We will continue to improve the relevant content.


## Framework Overview
![Alt text](https://github.com/Glory929/code/blob/main/img/network.png)

## Experimental results
### Table 1 Impact of component removal on network performance.
|Networks  | GBA  | IBC  | Up/Down-IBC  | HLFF  | ET(%)  | TC(%)  | WT(%)  | AVG(%)  
 |---- | ----- | ------ | ----- | ------ | ----- | ------  | ------  | ------ 
 |baseline | ×  | × | × | × | 83.90 | 88.00 | 91.00 | 87.62 
 | | ×  |  |  |  | 86.53 | 88.65 | 92.00 | 89.06
 | |   | × |  |  | 86.98 | 89.79 | 92.39 | 89.72
 | |   |  | × |  | 87.10 | 89.58 | 91.95 | 89.55 
 | |   |  |  | × | 86.90 | 89.65 | 92.13 | 89.56 
 | | ×  | × |  |  | 85.76 | 88.81 | 92.11 | 88.90
 | |   |  | × | × | 87.07 | 89.68 | 92.64 | 89.80 
 |ours |   |  |  |  | 87.42 | 90.70 | 92.31 | 90.15

 ### Table 2 Impact of hyperparameter settings on network performance."Lr" stands for the base learning rate, and "R" represents the channel scaling factor in the bottleneck block.

 Hyperparameter  | ET(%)  | TC(%)  | WT(%)  | AVG(%)
 ---- | ----- | ------   | ----- | ------
 Lr=0.002  | 87.14 | 90.32   | 92.48 | 89.99
 Lr=0.004  | 87.42 | 90.70 | 92.31 | 90.15  
 Lr=0.008  | 86.98 | 89.33   | 92.15 | 89.49
 R=1  | 86.28 | 89.19   | 92.03 | 89.17 
 R=2  | 87.42 | 90.70 | 92.31 | 90.15
 R=4  | 87.17 | 90.37   | 92.41 | 89.99
 




