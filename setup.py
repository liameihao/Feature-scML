from setuptools import setup
from Feature_scML.__init__ import __version__


setup(name="Feature_scML",
      version=__version__,
      url='https://github.com/liameihao/Feature-scML',
      author='liameihao',
      author_email='awd5174119@126.com',
      license='BSD 2-Clause',
      packages=['Feature_scML'],
      install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'seaborn',
        'matplotlib',
        'mlxtend==0.18.0',
        'scikit-learn==0.24.2',
        'minepy==1.2.3',
        'statsmodels==0.12.2',
        'skrebate==0.62',
        'shap==0.39.0',
        ],
      entry_points={
        'console_scripts': [
          'Feature_scML=Feature_scML.__main__:main'
          ]
        },
      python_requires=">=3.7",
      include_package_data=True,
      zip_safe=True
      )
