# AMBOKD
PyTorch code and the ESSVP dataset for our TPAMI 2024 paper "Adaptive Modality Balanced Online Knowledge Distillation for Brain-Eye-Computer based Dim Object Detection in Aerial Images".

# AMBOKD_ESSVP
This folder contains the ESSVP multimodal dataset and the experimental code for the AMBOKD-related algorithms
To run the experiment:
1. Download the essvp_dataset and pretrained uni-modal models from https://www.kaggle.com/datasets/lizixing23/essvp-dataset, and move them to the ESSVP_dataset and the models folders, respectively.
2. Run the following command in the AMBOKD_ESSVP folder to train the model and save the 10-fold cross-validation results:
   CMM: python main.py -- 
