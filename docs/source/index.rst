.. _index:

======================================
Welcome to Feature-scML documentation!
======================================

.. toctree::
   :maxdepth: 2
   :hidden:

   usage/fs
   usage/train
   usage/automl
   usage/plot

=======
Install
=======

.. code-block:: bash

   # create environment
   conda create -n env -f environment.yml
   # or
   conda env create -f environment.yml
   # activate environment
   conda activate Feature_scML

   #Install Feature-scML
   pip install git+https://github.com/liameihao/Feature-scML.git@main -U
   # or
   git clone https://github.com/liameihao/Feature-scML.git
   cd Feature-scML
   python setup.py install


=============
Documentation
=============
* :ref:`fs` -- Feature ranking.
* :ref:`train` -- Training Machine Learning models.
* :ref:`automl` -- Auto Machine learning training.
* :ref:`plot` -- Plot picture.


=======================
Data Format Requirement
=======================
The data format requires csv format. The first column is the sample label (Pay attention to the case of the **Label**.

+-------+-----------+-----------+-----+
| Label | feature 1 | feature 2 | ... |
+-------+-----------+-----------+-----+
| 1     | 2.0       | 3.0       | ... |
+-------+-----------+-----------+-----+
| 2     | 2.0       | 3.0       | ... |
+-------+-----------+-----------+-----+
| 1     | 50.0      | 80.0      | ... |
+-------+-----------+-----------+-----+
| 3     | 30.1      | 40.56     | ... |
+-------+-----------+-----------+-----+

=========
Reference
=========
* **F-score**: Chen, Yi-Wei, and Chih-Jen Lin. "Combining SVMs with various feature selection strategies." Feature extraction. Springer, Berlin, Heidelberg, 2006. 315-324.
* **CV2**: Brennecke, Philip, et al. "Accounting for technical noise in single-cell RNA-seq experiments." Nature methods 10.11 (2013): 1093-1095.
* **MIC**: Albanese, Davide, et al. "Minerva and minepy: a C engine for the MINE suite and its R, Python and MATLAB wrappers." Bioinformatics 29.3 (2013): 407-408.
* **TuRF**: Urbanowicz, Ryan J., et al. "Benchmarking relief-based feature selection methods for bioinformatics data mining." Journal of biomedical informatics 85 (2018): 168-188.
* **shap**: Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Proceedings of the 31st international conference on neural information processing systems. 2017.
* **Other methods** : Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." the Journal of machine Learning research 12 (2011): 2825-2830.
