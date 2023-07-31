# Automated Detection of Melanoma using Supervised Contrastive Learning

This repository contains code for building a melanoma classification model trained using supervised contrastive learning. The encoders which can be used in the model architecture include the Vision Transformer, ResNet50V2 and InceptionV3. Code is also provided that allows the user to pretrain their chosen encoder architecture using an image denoising task. Utilising a custom pre-training task is preferred to transfer learning in this context because the pre-trained weights available share little to no similarity to the dermatology images considered in this study, therefore pre-training on a set of images which are more domain specific yields a more valuable melanoma detection model.

Upon completion of the project, the README will be updated to include instructions for setting up a suitable working environment and for running the code.
