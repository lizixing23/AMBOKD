# MMOKD-DMEM
PyTorch code and the ESSVP dataset for our TPAMI 2024 paper "Multimodal Online Knowledge Distillation with Dynamic Modality Equilibrium Modulation for Dim Object Detection in Aerial Images".

# ESSVP dataset
This dataset is constructed by our brain-eye-computer based object detection system. There are image data in folder image and EEG data in folder EEG. To combine the image-EEG data pair, we need to use the script "Data_fusion_test_0229.py", which is able to process the experiment data from folder EEG and preprocess the multimodal data.

# Code for MMOKD-DMEM
The experiment codes for ESSVP dataset are include in folder experiment_ESSVP, the experiment codes for CIFAR-100 are include in folder experiment_CIFAR. 
