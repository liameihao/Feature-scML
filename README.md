<<<<<<< HEAD
# Feature-scML
### Welcome
 ```bash
conda create -n env -f environment.yml
# or
conda env create -f environment.yml
# install packages
python setup.py install
```

### Doc
https://Feature-scML.readthedocs.io/en/latest/index.html
=======
# Feature-scML
[![Documentation Status](https://readthedocs.org/projects/feature-scml/badge/?version=latest)](https://feature-scml.readthedocs.io/en/latest/?badge=latest)

### Install

It is recommended to install [conda](https://conda.io/en/latest/miniconda.html) to install the environment to avoid additional errors.

**Install conda**

 ```bash
conda create -n env -f environment.yml
# or
conda env create -f environment.yml

# activate environment
conda activate Feature_scML
 ```

**Install Feature-scML**

```bash
pip install git+https://github.com/liameihao/Feature-scML.git@main -U
# or
git clone https://github.com/liameihao/Feature-scML.git
cd Feature-scML
python setup.py install
```



tips: minepy may report errors, please install it separately

Solution

```bash
conda create -n env_name python=3.7
pip install minepy==1.2.3
# pip install statsmodels==0.12.2
# pip install skrebate==0.62
# pip install shap==0.39.0
```



### Doc

https://Feature-scML.readthedocs.io/en/latest/index.html
>>>>>>> 8bd37314557f03bbefbaf089a687d1810733c30f
