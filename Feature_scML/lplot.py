import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
import shap
from joblib import load
from .feature_selection import scale_data, cor_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE


# args
# Incremental feature selection (IFS) curves
def IFS_args(args, subparsers):
    parsers = subparsers.add_parser('IFS', prog='IFS', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    IFS_curve(data, filename, args.format)

# Feature Importance
def FW_args(args, subparsers):
    parsers = subparsers.add_parser('FW', prog='FW', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        '-n', '--number', 
        default=50,
        type=int,
        help='feature number'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    FW_curve(data, filename, args.format, args.number)

# Feature waterfall based on shap values
def fc_args(args, subparsers):
    parsers = subparsers.add_parser('waterfall', prog='waterfall', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        '--model_path',
        required=True,
        help='Model Path'
    )
    parsers.add_argument(
        '--method', '-m',
        choices=['svm'],
        default="svm",
        help='classifier (default=svm)'
    )
    parsers.add_argument(
        '-s', '--sample_id', 
        required=True,
        type=int,
        help='simple sample id'
    )
    parsers.add_argument(
        '-n', '--feature_number', 
        required=True,
        type=int,
        help='The number of features is consistent with the model'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("running...")
    print("."*20)
    feature_contribute(data, filename, 
                       args.model_path, 
                       args.format, 
                       args.sample_id, 
                       args.feature_number)
    print("Done")
    
    
def SHAP_args(args, subparsers):
    parsers = subparsers.add_parser('SHAP', prog='SHAP', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        '--model_path',
        required=True,
        help='Model Path'
    )
    parsers.add_argument(
        '--classifier', '-c',
        choices=['svm','rf','lr'],
        required=True,
        help='classifier type'
    )
    parsers.add_argument(
        '-n', '--feature_number', 
        required=True,
        type=int,
        help='The number of features is consistent with the model'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("running...")
    print("."*20)
    shap_sum(data, filename, 
             args.feature_number,
             args.model_path,
             args.format,
             args.classifier)
    print("Done")

# Feature beeswarm based on shap values of specific category
def fs_args(args, subparsers):
    parsers = subparsers.add_parser('beeswarm', prog='beeswarm', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        '--model_path',
        required=True,
        help='Model Path'
    )
    parsers.add_argument(
        '--method', '-m',
        choices=['svm'],
        default="svm",
        help='classifier (default=svm)'
    )
    parsers.add_argument(
        '-n', '--feature_number', 
        required=True,
        type=int,
        help='The number of features is consistent with the model'
    )
    parsers.add_argument(
        '-s', '--sample_label', 
        required=True,
        type=int,
        help='sample id is your label category to(0, 1, ...)'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("running...")
    print("."*20)
    feature_summary(data, filename,
                    args.feature_number, 
                    args.model_path, 
                    args.format, 
                    args.sample_label)
    print("Done")

# Principal Component Analysis (PCA)
def PCA_args(args, subparsers):
    parsers = subparsers.add_parser('PCA', prog='PCA', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        "-n", "--feature_number",
        required=True,
        type=int,
        help='Feature number'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("running...")
    print("."*20)
    PCA_plot(data,args.feature_number,2,args.format,filename)
    print("Done")


# T-SNE
def TSNE_args(args, subparsers):
    parsers = subparsers.add_parser('TSNE', prog='TSNE', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        "-n", "--feature_number",
        required=True,
        type=int,
        help='Feature number'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("running...")
    print("."*20)
    TSNE_plot(data,args.feature_number,2,args.format,filename)
    print("Done")

#  Confusion matrix
def CM_args(args, subparsers):
    parsers = subparsers.add_parser('CM', prog='CM', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        "-n", "--feature_number",
        required=True,
        type=int,
        help='The number of features is consistent with the model'
    )
    parsers.add_argument(
        '--model_path',
        required=True,
        help='Model Path'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    if args.output:
        os.chdir(args.output)
    print("running CM ...")
    print("."*20)
    CM_plot(data, filename, args.feature_number, args.model_path, args.format)
    print("Done")

#
def COR_args(args, subparsers):
    parsers = subparsers.add_parser('COR', prog='COR', usage='%(prog)s ')
    parsers.add_argument(
        "-i", '--input',
        required=True,
        help='Input data'
    )
    parsers.add_argument(
        '-m', '--method',
        choices=['pearson',"spearman","kendall"],
        required=True,
        help='correlation measure'
    )
    parsers.add_argument(
        '--format', '-f',
        choices=['png',"pdf"],
        default="png",
        help='Picture format(default=png)'
    )
    parsers.add_argument(
        "-n", "--feature_number",
        required=True,
        type=int,
        help='Feature number'
    )
    parsers.add_argument(
        '-o', '--output', 
        default=None,
        help='Output directory'
    )
    args = parsers.parse_args(args)
    filename = os.path.split(args.input)[1].split(".")[0]
    data = pd.read_csv(args.input, header=0)
    input_data = data.iloc[:,1:args.feature_number+1]
    if args.output:
        os.chdir(args.output)
    # cal cor
    print("running COR_ ...")
    correlation = cor_matrix(input_data.values, args.method)
    print("."*20)
    sns.heatmap(correlation,
                annot=True, 
                fmt=".2",
                cmap="Blues",
                xticklabels=input_data.columns,
                yticklabels=input_data.columns)
  
    plt.savefig("{0}_correlation_{1}_{2}.{3}".format(args.method, 
                                                     filename,
                                                     args.feature_number,
                                                     args.format),
                format=args.format,
                bbox_inches = 'tight')
    print("Done")

    
# main func   
def IFS_curve(data, filename, image_format):
    plt.figure(figsize=(5, 5))
    plt.plot(data['feature number'], data['train_accuracy'], label="train_accuracy")
    train_max = np.argmax(data['train_accuracy'])
    plt.scatter(data['feature number'][train_max], data['train_accuracy'][train_max], s=50, c="red", marker=(5, 1))
    test_max = np.argmax(data['test_accuracy'])
    plt.plot(data['feature number'], data['test_accuracy'], label="test_accuracy")
    plt.scatter(data['feature number'][test_max], data['test_accuracy'][test_max], s=50, c="red", marker=(5, 1))
    plt.xlabel("feature number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("{}_IFS.{}".format(filename, image_format), 
                format=image_format)
    

def lpolt_scale(X):
    X_std = (X - min(X)) / (max(X) - min(X))
    return X_std * 1


def FW_curve(data, filename, image_format, number):
    data_scale = lpolt_scale(data.iloc[:,1].values)
    size = number/5
    plt.figure(figsize=(10, size+2))
    plt.barh(data.iloc[:number,0][::-1], data_scale[:number][::-1])
    plt.ylim(-1,number)
    plt.xlabel('Feature Weights')
    plt.ylabel('Features')
    plt.savefig("{}_FeatureWeight.{}".format(filename, image_format), 
                format=image_format,
                bbox_inches = 'tight')
    

def feature_contribute(data, filename, model_path, image_format, sample_id, feature_number):
    model = load(model_path)
    X = scale_data(data.iloc[:,1:feature_number+1].values)
    y = data.iloc[:,0].values
    # svm shap
    f = lambda x: model.predict_proba(x)[:,y[sample_id]]
    # lr shap
    
    # rf shap
    
    # gnb
    
    explainer = shap.Explainer(f,X)
    shap_values = explainer(X[sample_id].reshape(1,-1))
    # plot
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("{}_simple_feature_contribute.{}".format(filename, image_format),
                format=image_format,
                bbox_inches = 'tight')
    plt.close()


def feature_summary(data, filename, feature_number, model_path, image_format, sample_label):
    model = load(model_path)
    X = scale_data(data.iloc[:,1:feature_number+1].values)
    y = data.iloc[:,0].values
    #
    X = X[y==sample_label]
    y = y[y==sample_label]
    # 
    f = lambda x: model.predict_proba(x)[:,sample_label]
    explainer = shap.Explainer(f,X)
    shap_values = explainer(X)
    shap_values.feature_names = data.columns[1:].values
    # plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("{}_simple_feature_summary.{}".format(filename, image_format),
                format=image_format,
                bbox_inches = 'tight')
    plt.close()

def shap_expain_choose(X, model, classifier_type):
    if classifier_type == "lr":
        explainer = shap.LinearExplainer(model, X)
        return explainer.shap_values(X)
    elif classifier_type == "svm":
        explainer = shap.KernelExplainer(model.predict_proba, X)
        return explainer.shap_values(X)
    elif classifier_type == "rf":
        explainer = shap.TreeExplainer(model, X)
        return explainer.shap_values(X, check_additivity=False)

   
def shap_sum(data, filename, feature_number, model_path, image_format, classifier_type):
    model = load(model_path).best_estimator_
    X = scale_data(data.iloc[:,1:feature_number+1].values)
    # 
    shap_values = shap_expain_choose(X, model, classifier_type)
    # plot
    shap.summary_plot(shap_values, X, plot_type="bar",
                      feature_names=data.columns[1:].values,
                      color=cm.Paired, show=False)
    plt.savefig("{}_SHAP_feature_summary.{}".format(filename, image_format),
                format=image_format,
                bbox_inches = 'tight')
    plt.close()


def PCA_plot(data, feature_number, lw, image_format, filename):
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:,1:feature_number+1].values)
    y = data.iloc[:,0].values
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)
    for i in np.unique(y):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=cm.Set3(int(i)), 
                    alpha=.8, lw=0, label=i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA')
    plt.savefig("{}_{}_PCA.{}".format(filename, feature_number, image_format),
                format=image_format,
                bbox_inches = 'tight')
    plt.close()
    

def CM_plot(data, filename, feature_number, model_path, image_format):
    model = load(model_path)
    X = scale_data(data.iloc[:,1:feature_number+1].values)
    y = data.iloc[:,0].values
    y_p = model.predict(X)
    #
    CM = confusion_matrix(y, y_p)
    
    CM_mean = CM / CM.sum(axis=1)
    plt.figure(figsize=(8, 8))
    classes = np.unique(y)
    plt.imshow(CM_mean, cmap=plt.cm.Blues)
    plt.colorbar(shrink=0.8)
    #
    indices = range(len(CM))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.xlabel('Predicted Label',{'size':12})
    plt.ylabel('True Label',{'size':12})
    for first_index in range(len(CM)):
        for second_index in range(len(CM[first_index])):
            plt.text(second_index, first_index, CM[first_index][second_index],
                    va='center', ha='center')
    plt.savefig("confusion_matrix_{}_{}.{}".format(filename, feature_number, image_format), 
                format=image_format, 
                bbox_inches = 'tight')
    print("Done")


def TSNE_plot(data, feature_number, lw, image_format, filename): 
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:,1:feature_number+1].values)
    y = data.iloc[:,0].values
    pca = PCA(n_components=15)
    tsne = TSNE(n_components=2, init='pca')
    X_pca = pca.fit_transform(X)
    X_embedded = tsne.fit_transform(X_pca)
    for i in np.unique(y):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], 
                    alpha=.8, lw=0, label=int(i))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('T-SNE')
    plt.savefig("{}_{}_T-SNE.{}".format(filename, feature_number, image_format),
                format=image_format,
                bbox_inches = 'tight')
    plt.close()

    
__all__ = ['IFS_args', 'FW_args', 'fc_args', 
           'fs_args', 'PCA_args',"TSNE_args",
           'CM_args', 'COR_args',"SHAP_args"]




# def scale_data(X_train):
#     min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = min_max_scaler.fit(X_train)
#     X_train_ = scaler.transform(X_train)
#     return X_train_

# data = pd.read_csv("test_123/train_pca_data.csv")
# filename = "test"
# model_path = "test_123/train_pca_data_svm.joblib"
# image_format = "png"
# sample_label = 1


