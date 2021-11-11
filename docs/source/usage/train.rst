.. _train:

================
Machine learning 
================

Feature-scML cls func
---------------
cls module is Machine Learning classification function, including Support Vector Machine, Random Forest, Gaussian Naive Bayes, and Logical Regression.
Input data and test data is required CSV format. We integrated the hyperparameter optimization function in the training process. 
Test data is not required. IF test data is not input, the train data wil be splited into train data and test data (train:test=8:2).

.. code-block:: bash
    
    $Feature-scML cls -h
    usage: cls

    optional arguments:
    -h, --help            show this help message and exit
    -c {lr,svm,rf,gnb}, --classifier {lr,svm,rf,gnb}
                            Select a machine learning method:
                            lr (Logical Regression)
                            svm (Support Vector Machine)
                            rf (Random Forest)
                            gnb (Gaussian Naive Bayes)
    -i INPUT_TRAIN, --input_train INPUT_TRAIN
                            Input train data (CSV)
    -o OUTPUT, --output OUTPUT
                            Output directory
    --njobs NJOBS         The number of jobs to run in parallel
    --input_test INPUT_TEST
                            Input test data filename path (CSV)
    --getmodel GETMODEL   Generate model files (default=False)



Command
-------

+--------------------+------------------+----------------------------------------------+
| Parameters         | optional         | Descripton                                   |
+====================+==================+==============================================+
|| ---classifier, -c || lr,svm,rf,gnb   || - lr (Logical Regression)                   |
||                   ||                 || - svm (Support Vector Machine)              |
||                   ||                 || - rf (Random Forest)                        |
||                   ||                 || - gnb (Gaussian Naive Bayes)                |
+--------------------+------------------+----------------------------------------------+
| ---input_train,-i  | filename path    | input filename path (CSV format)             |
+--------------------+------------------+----------------------------------------------+
| ---output, -o      | output directory | output directory (default:Current directory) |
+--------------------+------------------+----------------------------------------------+
| ---njobs           | int, default=1   | The number of jobs to run in parallel        |
+--------------------+------------------+----------------------------------------------+
| ---input_test      | filename path    | If None, train dataset will be splited       |
+--------------------+------------------+----------------------------------------------+
| ---getmodel        | True or False    | If True, model file will be saved            |
+--------------------+------------------+----------------------------------------------+


Example
-------

.. code-block:: bash
    
    $Feature-scML cls -i example.csv -c svm
    Feature selection: cv2
    cv2 is running
    ....................
    The identity link function does not respect the domain of the Gamma family.
    cv2: DONE
    $ls
    example.csv  example_cv2.csv  example_cv2_data.csv
    # example_cv2.csv is feature importance of cv2 method
    # The colnames of example_cv2_data.csv are sorted by feature ranking of cv2 method.


**example_cv2.csv**

The result will generate a dataframe with column names of feature and score.

+---------+--------------------+
| Feature | cv2_score          |
+---------+--------------------+
| CCKBR   | 122.89836751624631 |
+---------+--------------------+
|| IFITM1 || 15.62471687317092 |
|| ...    || ...               |
+---------+--------------------+
| PDCL3   | -90.00136155460666 |
+---------+--------------------+


**example_cv2_data.csv**

+-------+--------------------+--------------------+-----+---------------+
| Label | CCKBR              | IFITM1             | ... | PDCL3         |
+-------+--------------------+--------------------+-----+---------------+
| 6     | 0.7922727272727272 | 0.7636363636363637 | ... | 93.6201131016 |
+-------+--------------------+--------------------+-----+---------------+
| 3     | 0.8276233766233766 | 0.7772727272727272 | ... | 1517.12046654 |
+-------+--------------------+--------------------+-----+---------------+
| ...   | ...                | ...                | ... | ...           |
+-------+--------------------+--------------------+-----+---------------+
| 3     | 0.0                | 121.252831887      | ... | 1234.49979645 |
+-------+--------------------+--------------------+-----+---------------+

