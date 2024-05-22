# MMOKD-DMEM
PyTorch code and the ESSVP dataset for our TPAMI 2024 paper "Multimodal Online Knowledge Distillation with Dynamic Modality Equilibrium Modulation for Dim Object Detection in Aerial Images".

# ESSVP dataset
This dataset is created by our brain-eye-computer based object recognition system. There are image data in the image folder and EEG data in the EEG folder. To fuse the image-EEG data pair, we need to use the script "Data_fusion_test_0229.py", which is able to process the experiment data from the EEG folder and pre-process the multimodal data.

# Code for MMOKD-DMEM
The experiment codes for ESSVP dataset are include in folder experiment_ESSVP, the experiment codes for CIFAR-100 are include in folder experiment_CIFAR. 
