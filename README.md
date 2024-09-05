# SIMCLR-
“run.py” script trains a model using the SimCLR framework.

”SimCLR.py”: Defines the main training logic for the SimCLR model.

To test the model's performance, use the ”simclr1.py”. Note: if you want to test the model's performance, you need to comment out the following line in run.py:

from simclr import SimCLR 

and uncomment the following line:

#from simclr1 import SimCLR

Also, change the dataset path in run.py to the dataset you want to use for performance testing.
