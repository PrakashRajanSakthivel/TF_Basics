# TF_Basics
Tensor Flow Basics using Python.

Machine - windows 10 <br />
Python version - 3.7.0 <br />
TF Version - 1.14.0


Issues faced: 

1. After installing tensorflow, while running the code was getting the error 
"Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2". 
Solution:  https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u -- solution by Z.Wei worked

2. If facing issue in loading the url in tensorboard, then load specify the url and port.
  tensorboard --logdir="entire\folder\path" --host localhost --port 8085(anyport)

 


* This was created while learning tensorflow fromr PluralSight
