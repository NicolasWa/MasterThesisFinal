# Master Thesis: "Detection of tumor area in histological sections of breast cancer using deep learning"
Breast cancer tumor segmentation using Deep Learning (DL) on Ki67 Whole Slide Images (WSIs) coming from a private data set. 
Nearly all research on tumor segmentation using DL on WSIs had been done using the Hematoxylin and Eosin (H&E) staining, and this work was investigating what results can be achieved when using WSIs stained with the Ki67 proliferation instead.

Short summary of the pipeline:
- Data: Ki67 stained WSIs of the order of the GB each, private data set, annotated by a pathologist
- Data preprocessing: dimensionality reduction, tissue extraction, sliding-window (with and without overlap, and for different tile sizes), data augmentation
- Model: DL model based on the U-Net architecture (convolutional network with encoder + decoder)
- Post-processing: stitching of the tiles' predictions with and without overlap, threshold

Several parameters were studied, namely the tradeoff between the context (tile size) and the resolution (magnification), and whether or not using overlapping patches as another data augmentation technique or as a postprocessing technique lead to any improvement.

[For more information, click here to read the master thesis](Master_Thesis.pdf)

