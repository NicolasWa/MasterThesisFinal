# MasterThesisFinal
Breast cancer tumor detection using Deep Learning (U-Net architecture) on Ki67 WSIs coming from a private data set.
[Click here to read the master thesis](Master_Thesis.pdf)
The data set first needs to be decomposed into tiles (cf data_prep.py)
Make sure that the right folders for the data set exist on your computer at similar paths than in the code
To launch an experiment, fill in the parameters in the experiments.py file and launch it.
To visualize the results, type tensorboard --logdir tensorboard in the current folder terminal
To evaluate the models, see Evaluator.py

