.. _fs:

=================
Feature Selection
=================


fs_cls fs func
--------------

fs module is feature selection function, including F-score algorithm (**fscore**), 
the squared coefficient of variation (**cv2**), 
principal component analysis (**pca**), random forest classifier (**rfc**), ANOVA Pvalue (**ano**), 
the maximal information coefficient measure (**mic**), 
ReliefF (**turf**), and linear support vector machine (**linearsvm**). 
These methods are referred in the :ref:`index <Reference>`.


.. code-block:: bash

    $fs_cls fs -h
    usage: fs

    optional arguments:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                            Input train data (CSV)
    -m {fscore,pca,cv2,rfc,ano,mic,turf,linearsvm}, --method {fscore,pca,cv2,rfc,ano,mic,turf,linearsvm}
                            Feature selection method
    -o OUTPUT, --output OUTPUT
                            Output directory
    --njobs NJOBS         The number of jobs to run in parallel (Only turf,
                            default=1)




Command
-------

+---------------+--------------------+----------------------------------------------+
| Parameters    | optional           | Descripton                                   |
+===============+====================+==============================================+
|| --method, -m || fscore, pca, cv2, || - fscore (F-score algorithm)                |
||              || rfc, ano, mic,    || - pca (principal component analysis)        |
||              || turf, linearsvm   || - cv2 (squared coefficient of variation)    |
||              ||                   || - rfc (random forest classifier)            |
||              ||                   || - ano (ANOVA Pvalue)                        |
||              ||                   || - mic (the maximal information coefficient) |
||              ||                   || - turf (ReliefF)                            |
||              ||                   || - linearsvm (linear support vector machine) |
+---------------+--------------------+----------------------------------------------+
| --input,-i    | filename path      | input filename path (CSV format)             |
+---------------+--------------------+----------------------------------------------+
| --output, -o  | output directory   | output directory (default:Current directory) |
+---------------+--------------------+----------------------------------------------+
| --njobs       | int, default=1     | The number of jobs to run in parallel        |
+---------------+--------------------+----------------------------------------------+

Example
-------

.. code-block:: bash

    # default start, end, and step 
    $fs_cls fs -i example.csv -m cv2 


The result will generate two dataframe. As follow:

**example_cv2.csv**

+---------+-----------+
| Feature | cv2_score |
+---------+-----------+
| CCKBR   | 122.898   |
+---------+-----------+
| IFITM1  | 15.624    |
+---------+-----------+
| CLDN10  | 11.051    |
+---------+-----------+
| ...     | ...       |
+---------+-----------+

The dataframe sorted by the features of the original data.

**example_cv2_data.csv**

+-------+--------+--------+-----+
| Label | CCKBR  | IFITM1 | ... |
+-------+--------+--------+-----+
| 6     | 229.00 | 32.21  | ... |
+-------+--------+--------+-----+
| 3     | 25.14  | 76.82  | ... |
+-------+--------+--------+-----+
| 6     | 154.65 | 52.62  | ... |
+-------+--------+--------+-----+