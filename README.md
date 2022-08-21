# MasterThesisFinal
Breast cancer tumor detection using DL on Ki67 WSIs
Pipeline inspired on the code of Adrien Foucart https://github.com/adfoucart/dlia-videos

The data set first needs to be decomposed into tiles (cf data_prep.py)
Make sure that the right folders for the data set exist on your computer at similar paths than in the code
To launch an experiment, fill in the parameters in the experiments.py file and launch it.
To visualize the results, type tensorboard --logdir tensorboard in the current folder terminal
To evaluate the models, see Evaluator.py

