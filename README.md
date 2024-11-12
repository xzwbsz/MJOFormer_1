# MJOFormer

### 1. Configure the environment

python == 3.8.0 pytorch == 1.10.1

pip install -r requirement.txt **(some required dependencies)**

### 2. Dataset preparation

(For inference) For our model, we input the day-by-day data of ERA5 for the first **7** days and inference the RMM index for the second **35** days; for the ERA5 data, we need to perform data preprocessing.

(For training) we utilized the ECMWF data and CMA data from the S2S database.

(1) The ERA5 data include SST, OLR (ttr), U200, and U850, ranging from -15° to 15° and 0° to 360°; the precision is 2.5°;

(2) The data were first subtracted from the interannual cycle (1991 to 2010)

(3) Subtract the first 120 days of averaging (ENSO)

(4) Normalize the data

(5) Sliding window to construct more data sets

Steps 2 to 5 can be found in the code: create_re_dataset_for7_7_35.py

After processing, the shape of our input data, i.e. X_test, is.

**[total sample size, time length(7), latitude(13), longitude(144), number of variables(4)]**

Our output RMM index, i.e. the shape of Y_test is:

**[total sample size, length of time(35), RMM index(2)]**

### 3. Inference

We encapsulate the model framework in **main.py**, while some related computational methods, such as the computation of cor and rmse, are encapsulated in **function.py**. Finally, the testing of the model is in the **test** function of **main**.

Since we are providing the trained model, we need to import the relevant weight parameters, i.e. **rmm1.pth** and **rmm2.pth**.

Some relevant paths need to be changed manually in .py.

### 4. Overview

The framework of MJOFormer is as follows：

![framework](framework.png)



### Note

Please forgive me for not being able to upload the data to GitHub as it is too large for ERA5 and S2S. However, the above has given for the processing method.
