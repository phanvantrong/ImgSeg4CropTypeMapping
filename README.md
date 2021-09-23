# CropTypeMappingUNET
Corn and Soybean Mapping model at 30 m resolution from Landsat 8/ Sentinel 2

**train.py** -  trains the model. Make sure the data directory/folder are set correctly. The training data can be exported as _TFRecord_ from Earth Engine. The model expects data as 256x256 samples with the bands and the labels stacked together as channels. (Will upload the export code in python through earth engine api later)
**model.py** - The UNET image segmentation model.
**confusion_matrix.py** - includes helper functions to save a confusion matrix image as part of the logs that can be monitored via tensorboard.

