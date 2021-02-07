# BYOL-Pytorch

Pytorch Implementation for BYOL (Verified on Imagenet)

This code has been verified on Imagenet, and get similar performance with the result reported in the original paper. we train the model on 32 V100 GPU (32G), it roughly takes 7 to 8 mins per epoch.

To run the code, please change the Dataset setting (dataset/Imagenet.py), and Pytorch DDP setting (util/dist_init.py) for your own server enviroments.

With Batch size = 4096, we get the following result

Results on Imagenet:
|          |Arch | BatchSize | Epochs | Accuracy (our reproduce)|  Accuracy (paper) |
|----------|:----:|:---:|:---:|:---:|:---:|
|  BYOL | ResNet50 | 4096 | 100  |  68.8 % | - |
|  BYOL | ResNet50 | 4096 | 200  |  72.7 % | 72.5% |
