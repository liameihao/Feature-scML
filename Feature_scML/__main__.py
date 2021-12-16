import argparse
from argparse import RawTextHelpFormatter
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
import textwrap
#
from .IFS import overall_predict
from .feature_selection import fs_output
from .models import model_train
from .lplot import *


# warnings.filterwarnings("ignore")


def fs(args, subparsers):
    parsers = subparsers.add_parser('fs', prog='fs', usage='%(prog)s ')
    parsers.add_argument(
        '-i', '--input', 
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '-m', '--method', 
        required=True,
        choices=['fscore', 'pca', 'cv2', 'rfc',
                 'mic', 'turf', 'linearsvm'],
        help='Feature selection method'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    parsers.add_argument(
        '--njobs', 
        default=1, 
        type=int,
        help='The number of jobs to run in parallel (Only turf, default=1)'
    )
    
    #
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("Feature selection: {}".format(args.method))
    print("{} is running".format(args.method))
    print("."*20)
    fs_output(data, args.method, filename, args.njobs)
    print("{}: DONE".format(args.method))


def cls(args, subparsers):
    parsers = subparsers.add_parser(
        'cls', 
        prog='cls', 
        usage='%(prog)s ',
        formatter_class=RawTextHelpFormatter
    )
    parsers.add_argument(
        '-c', '--classifier', 
        required=True,
        choices=['lr', 'svm', 'rf', 'gnb'],
        help=textwrap.dedent('''\
            Select a machine learning method: 
            lr (Logical Regression)
            svm (Support Vector Machine)
            rf (Random Forest)
            gnb (Gaussian Naive Bayes)''')
    )
    parsers.add_argument(
        '-i', '--input_train', 
        required=True,
        help="Input train data (CSV)"
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    parsers.add_argument(
        '--njobs', 
        default=1, 
        type=int,
        help='The number of jobs to run in parallel'
    )
    parsers.add_argument(
        "-t", '--input_test', 
        default=None,
        help='Input test data filename path (CSV)'
    )
    parsers.add_argument(
        '--getmodel', 
        default=False, 
        type=bool,
        help='Generate model files (default=False)'
    )
    #
    args = parsers.parse_args(args)
    filename = os.path.split(args.input_train)[1].split(".")[0]
    data = pd.read_csv(args.input_train, header=0)
    X_train = data.iloc[:, 1:].values
    y_train = data['Label'].values
    if not args.input_test: 
        # Split arrays into random train and test subsets
        print("Data is split into training set and test set according to 8:2")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
    else:
        test = pd.read_csv(args.input_test, header=0)
        X_test = test.iloc[:, 1:].values
        y_test = test['Label'].values
    if args.output:
        os.chdir(args.output)
    model_train(X_train, y_train, X_test, y_test,
                njob=args.njobs,
                get_model=args.getmodel,
                method=args.classifier,
                filename=filename)


def automl(args, subparsers):
    parsers = subparsers.add_parser(
        'automl', 
        prog='automl', 
        usage='%(prog)s ',
        formatter_class=RawTextHelpFormatter
    )
    parsers.add_argument(
        "-i", '--input_train',
        required=True,
        help='Input train data filename path (CSV)'
        )
    parsers.add_argument(
        "-t", '--input_test',
        default=None,
        help='Input test data filename path (CSV)'
        )
    parsers.add_argument(
        '--method', '-m',
        choices=['fscore', 'pca', 'cv2', 'rfc',
                 'mic', 'turf', 'linearsvm'],
        required=True,
        help='Select a feature selection method'
    )
    parsers.add_argument(
        '--disable_fs_method',
        default=False,
        type=bool,
        help='Use IFS directly without feature ranking (default: False)'
    )
    parsers.add_argument(
        "--start",
        type=int,
        default=10,
        help="Feature Number start (default=10)"
    )
    parsers.add_argument(
        "--end",
        type=int,
        default=None,
        help="Feature Number end (default=all features)"
    )
    parsers.add_argument(
        "--step",
        type=int,
        default=10,
        help="Feature Number step (default=10)"
    )
    parsers.add_argument(
        "--njobs",
        default=1,
        type=int,
        help="Number of jobs to run in parallel (default=1)"
    )
    parsers.add_argument(
        '--classifier', '-c',
        choices=['svm', 'rf', 'gnb', 'lr'],
        required=True,
        help=textwrap.dedent('''\
            Select a machine learning method: 
            lr (Logical Regression)
            svm (Support Vector Machine)
            rf (Random Forest)
            gnb (Gaussian Naive Bayes)''')
    )
    parsers.add_argument(
        '--getmodel', 
        default=False, 
        type=bool,
        help='Generate model files (default=False)'
    )
    parsers.add_argument(
        '-o', "--output",
        default=None,
        help='Output directory (default=current directory)'
    )
    args = parsers.parse_args(args)
    overall_predict(args)
    print("DONE!")
    

def main():
    parsers = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description='AUTOML'
    )
    subparsers = parsers.add_subparsers(help='sub-command help')
    # feature selection
    subparsers_fs = subparsers.add_parser(
        'fs',
        add_help=False,
        help="Feature selection function"
    )
    subparsers_fs.set_defaults(func=fs)
    # classification
    subparsers_cls = subparsers.add_parser(
        'cls',
        add_help=False,
        help="Machine learning function"
        )
    subparsers_cls.set_defaults(func=cls)
    # auto machine learning
    subparsers_automl = subparsers.add_parser(
        'automl',
        add_help=False,
        help="AUTO Machine learning"
    )
    subparsers_automl.set_defaults(func=automl)
    # lplot
    subparsers_lplot = subparsers.add_parser(
        'plot',
        help="PLOT"
    )
    subparsers_lplot_s = subparsers_lplot.add_subparsers(help='sub-command help')
    # IFS   
    subparsers_lplot_IFS = subparsers_lplot_s.add_parser(
        'IFS', 
        add_help=False,
        help='Incremental feature selection (IFS) curves')
    subparsers_lplot_IFS.set_defaults(func=IFS_args)
    # FW
    subparsers_lplot_FW = subparsers_lplot_s.add_parser(
        'FW', 
        add_help=False,
        help='Feature weights (FW)')
    subparsers_lplot_FW.set_defaults(func=FW_args)
    # waterfall
    subparsers_lplot_waterfall = subparsers_lplot_s.add_parser(
        'waterfall', 
        add_help=False,
        help='Feature waterfall based on shap values')
    subparsers_lplot_waterfall.set_defaults(func=fc_args)
    # beeswarm
    subparsers_lplot_beeswarm = subparsers_lplot_s.add_parser(
        'beeswarm', 
        add_help=False,
        help='Feature beeswarm based on shap values of specific category')
    subparsers_lplot_beeswarm.set_defaults(func=fs_args)
    #
    # SHAP
    subparsers_lplot_SHAP = subparsers_lplot_s.add_parser(
        'SHAP', 
        add_help=False,
        help='Feature bar plot based on shap value')
    subparsers_lplot_SHAP.set_defaults(func=SHAP_args)
    #
    # PCA
    subparsers_lplot_PCA = subparsers_lplot_s.add_parser(
        'PCA', 
        add_help=False,
        help='Principal Component Analysis (PCA)')
    subparsers_lplot_PCA.set_defaults(func=PCA_args)
    # TSNE
    subparsers_lplot_TSNE = subparsers_lplot_s.add_parser(
        'TSNE', 
        add_help=False,
        help='T-distributed Stochastic Neighbor Embedding(TSNE)')
    subparsers_lplot_TSNE.set_defaults(func=TSNE_args)
    #
    # Confusion matrix (CM)
    subparsers_lplot_CM = subparsers_lplot_s.add_parser(
        'CM', 
        add_help=False,
        help='Confusion matrix (CM)')
    subparsers_lplot_CM.set_defaults(func=CM_args)
    #
    # correlation (pearson, spearman, kendall)
    subparsers_lplot_COR = subparsers_lplot_s.add_parser(
        'cor', 
        add_help=False,
        help='Feature correlation')
    subparsers_lplot_COR.set_defaults(func=COR_args)
    #
    tmp, known_args = parsers.parse_known_args()
    if not known_args:
        print("Please use \"--help\" or \"-h\" ")
        return
    try:
        tmp.func(known_args, subparsers)
    except AttributeError:
        pass


if __name__ == '__main__':
    main()
