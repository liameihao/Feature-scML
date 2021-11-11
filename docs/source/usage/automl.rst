.. _automl:

=====================
Auto Machine Learning
=====================

Feature_scML automl func
------------------
automl is Auto Machine Learning, including fs module and cls module.
Input data and test data is required **CSV format**. We integrated the hyperparameter optimization function in the training process. 
Test data is not required. **If test data is not input, the train data wil be splited into train data and test data (train:test=8:2).**
In order to simplify the machine learning process, the module can automatically perform feature selection 
methods, and combine incremental features to train machine learning models. 
Finally, the user can obtain the feature data set ranked according to the feature selection method 
and the result table of incremental feature training.


.. code-block:: bash

    $Feature_scML automl -h
    usage: automl

    optional arguments:
    -h, --help            show this help message and exit
    -i INPUT_TRAIN, --input_train INPUT_TRAIN
                            Input train data (CSV)
    -t INPUT_TEST, --input_test INPUT_TEST
                            Input test data (CSV)
    --method {fscore,pca,cv2,rfc,ano,mic,turf,linearsvm}, -m {fscore,pca,cv2,rfc,ano,mic,turf,linearsvm}
                            Select a feature selection method
    --start START         Feature Number start (default=10)
    --end END             Feature Number end (default=all features)
    --step STEP           Feature Number step (default=10)
    --njobs NJOBS         Number of jobs to run in parallel (default=1)
    --classifier {svm,rf,gnb,lr}, -c {svm,rf,gnb,lr}
                            Select a machine learning method:
                            lr (Logical Regression)
                            svm (Support Vector Machine)
                            rf (Random Forest)
                            gnb (Gaussian Naive Bayes)
    --getmodel GETMODEL   Generate model files (default=False)
    -o OUTPUT, --output OUTPUT
                            Output directory (default=current directory)


Command
-------

+--------------------+---------------------------+----------------------------------------------+
| Parameters         | Optional                  | Descripton                                   |
+====================+===========================+==============================================+
| ---input_train,-i  | filename path             | input Train data filename path (CSV format)  |
+--------------------+---------------------------+----------------------------------------------+
| ---input_test,-t   | filename path             | input Test data filename path (CSV format)   |
+--------------------+---------------------------+----------------------------------------------+
|| ---method, -m     || fscore, pca, cv2,        || The details of the methods are in the       |
||                   || rfc, ano, mic,           || fs module                                   |
||                   || turf, linearsvm          ||                                             |
+--------------------+---------------------------+----------------------------------------------+
|| ---classifier, -c || lr,svm,rf,gnb            || - lr (Logical Regression)                   |
||                   ||                          || - svm (Support Vector Machine)              |
||                   ||                          || - rf (Random Forest)                        |
||                   ||                          || - gnb (Gaussian Naive Bayes)                |
+--------------------+---------------------------+----------------------------------------------+
| ---start           | int, default=10           | Minimal number of features                   |
+--------------------+---------------------------+----------------------------------------------+
| ---end             | int, default=all features | Maximum number of features                   |
+--------------------+---------------------------+----------------------------------------------+
| ---step            | int, default=10           | Step size of incremental feature training    |
+--------------------+---------------------------+----------------------------------------------+
| ---output, -o      | output directory          | output directory (default:Current directory) |
+--------------------+---------------------------+----------------------------------------------+
| ---njobs           | int, default=1            | The number of jobs to run in parallel        |
+--------------------+---------------------------+----------------------------------------------+
| ---getmodel        | True or False             | If True, model file will be saved            |
+--------------------+---------------------------+----------------------------------------------+

Example
-------

.. code-block:: bash

    # default start, end, and step 
    $Feature_scML automl -i example.csv -c svm -m cv2 
    The identity link function does not respect the domain of the Gamma family.
    feature number: 10
    train accuracy: 0.7923
    test_accuracy: 0.7636
    best parameters: {'C': 8192, 'gamma': 0.00048828125}
    feature number: 20
    train accuracy: 0.8276
    test_accuracy: 0.7773
    best parameters: {'C': 8, 'gamma': 0.5}
    ...
    feature number: 100
    train accuracy: 0.8824
    test_accuracy: 0.8682
    best parameters: {'C': 512, 'gamma': 0.001953125}
    DONE!


The result will generate a dataframe with column names of 
feature number, train_accuracy, test_accuracy and parameters (optimal hyperparameter).


+----------------+--------------------+--------------------+---------------------------------------+
| feature number | train_accuracy     | test_accuracy      | parameters                            |
+----------------+--------------------+--------------------+---------------------------------------+
| 10             | 0.7922727272727272 | 0.7636363636363637 | "{'C': 8192, 'gamma': 0.00048828125}" |
+----------------+--------------------+--------------------+---------------------------------------+
| 20             | 0.8276233766233766 | 0.7772727272727272 | "{'C': 8, 'gamma': 0.5}"              |
+----------------+--------------------+--------------------+---------------------------------------+
| ...            | ...                | ...                | ...                                   |
+----------------+--------------------+--------------------+---------------------------------------+
| 100            | 0.8824285714285715 | 0.8681818181818182 | "{'C': 512, 'gamma': 0.001953125}"    |
+----------------+--------------------+--------------------+---------------------------------------+


.. code-block:: bash

    # If output is None, model file will saved in current directory
    # example_lr.joblib is saved in current directory.
    # start = 20, step = 20, end = 60
    $Feature_scML automl -i example.csv -c svm -m cv2 --start 20 --step 20 --end 60 --njobs 20 --getmodel True
    $ls
    20-60_cv2_SVM_accuracy.csv  example.csv  example_40_svm.joblib  example_cv2.csv
    example_20_svm.joblib       example_60_svm.joblib  example_cv2_data.csv
