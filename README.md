# AMBOKD
This repository provides the official implementation of “Adaptive Modality Balanced Online Knowledge Distillation for Brain-Eye-Computer-Based Dim Object Detection,” accepted by IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2025. We introduce an adaptive, modality-balanced online knowledge distillation framework and reproduce the key experiments and results. The project also includes materials for our ESSVP dataset, a multimodal brain–eye–computer dim-object detection dataset collected for this work (with dataset description and usage instructions included in the repo). A demo video is available at https://youtu.be/Yj01V8TBlUQ and can also be downloaded from this repository. If you find this code or the dataset useful, please cite: 
Li, Zixing, et al. “Adaptive Modality Balanced Online Knowledge Distillation for Brain–Eye–Computer-Based Dim Object Detection.” IEEE Transactions on Neural Networks and Learning Systems, pp. 1-15, 2025.
# AMBOKD_ESSVP
This folder contains the ESSVP multimodal dataset and the experimental code for the AMBOKD-related algorithms
To run the experiment:
1. Download the essvp_dataset and pretrained uni-modal models from https://www.kaggle.com/datasets/lllll789/ambokd-data, and move them to the ESSVP_dataset and the models folders, respectively.
2. Run the following command in the AMBOKD_ESSVP folder to train the model and save the 10-fold cross-validation results:  
   CMM: python main.py --model_s_fusion CMM --distill None --IS_adjust_lr False --IS_adjust_kd False  
   MLB: python main.py --model_s_fusion MLB --distill None --IS_adjust_lr False --IS_adjust_kd False  
   AMM: python main.py --model_s_fusion AMM --distill None --IS_adjust_lr False --IS_adjust_kd False  
   E-KD: python main.py --model_s_fusion AMM --distill E-KD --IS_adjust_lr False --IS_adjust_kd False  
   V-KD: python main.py --model_s_fusion AMM --distill V-KD --IS_adjust_lr False --IS_adjust_kd False  
   MKD: python main.py --model_s_fusion AMM --distill MKD --IS_adjust_lr False --IS_adjust_kd False  
   EMKD: python main.py --model_s_fusion AMM --distill EMKD --IS_adjust_lr False --IS_adjust_kd False  
   CA-MKD: python main.py --model_s_fusion AMM --distill CA-MKD --IS_adjust_lr False --IS_adjust_kd False  
   DML: python main.py --model_s_fusion DML --distill DML --IS_adjust_lr False --IS_adjust_kd False  
   KDCL: python main.py --model_s_fusion KDCL --distill KDCL --IS_adjust_lr False --IS_adjust_kd False  
   EML: python main.py --model_s_fusion EML --distill EML --IS_adjust_lr False --IS_adjust_kd False  
   MMOKD: python main.py --model_s_fusion MMOKD --distill MMOKD --IS_adjust_lr False --IS_adjust_kd False  
   MMOKD-DG: python main.py --model_s_fusion MMOKD --distill MMOKD --IS_adjust_lr True --IS_adjust_kd False  
   AMBOKD: python main.py --model_s_fusion MMOKD --distill MMOKD --IS_adjust_lr True --IS_adjust_kd True

