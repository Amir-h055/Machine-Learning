README for G13
--------------
Group Members:
Annika Timermanis, 40131128
Axel Solano, 40046154
AmirHossein Hadinezhad, 

Our project relies mainly on:
- sklearn
- pandas
- matplotlib
- PyTorch 
- numpy
Import tabulate is required, used for table representation of performance of each model.

Our code submission contains the following files:

  ./code/requirements.txt    <-- python environment requirements.txt file
  ./code/util/*.py           <-- some utility functions we wrote
  ./code/preprocess.py       <-- script to preprocess our data
  ./code/train_svm.py        <-- script to train our SVM
  ./code/train_rf.py         <-- script to train our random forest
  ./code/train_nnet.py       <-- script to train our neural net
  ./code/report.ipynb        <-- notebook with figures and tables for report
  ./data/housing_prices.csv  <-- raw data set
  
* The files should be run in the order:
   
   1) Any order for the .ipynb files in Classification and Regression folders
   2) cnn.ipynb 
   3) Decision_tree.ipynb
   4) report.ipynb

* Training times vary, most models can be trained within 5-10 minutes. Datasets that caused all models training times to be very long to  are listed as follows:
- GPU kernel (Regression), +10 hours.
- Adult (Classification), +3 hours.
- Yeast (Classification), +4 hours.
- CNN (CIFAR), +10 hours

* Every .ipynb file from the Classification folder writes to an external file and stores all the trained models for classification.
* Every .ipynb file from the Regression folder, writes to an external file, and stores all the trained models for regression.
