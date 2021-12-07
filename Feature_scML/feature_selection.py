import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from skrebate import TuRF
from minepy import MINE
from scipy import stats


def fs_output(data, method_name, filename, njobs=1):
    label = data['Label'].values
    if method_name == "cv2":
        feature_score = cal_cv2(data)
    elif method_name == "pca":
        feature_score = cal_pca(data)
    elif method_name == "fscore":
        feature_score = fscore(data)
    elif method_name == "rfc":
        feature_score = feature_selection_rfc(data, label)
    elif method_name == "mic":
        feature_score = mic(data, label)
    elif method_name == "turf":
        feature_score = feature_selection_Turf(data, label, njobs)
    elif method_name == "linearsvm":
        feature_score = feature_selection_linear_svm(data, label)
    else:
        raise ValueError('Method input wrose!')
    score = pd.DataFrame({
        'Feature': data.columns[1:],
        "{}_score".format(method_name): feature_score
    })
    score = score.sort_values("{}_score".format(method_name), ascending=False)
    score.to_csv("{0}_{1}.csv".format(filename, method_name), index=False)
    data_train = data.reindex(['Label'] + list(score['Feature']), axis=1)
    data_train.to_csv("{0}_{1}_data.csv".format(filename, method_name),
                      index=None)
    return data_train


def scale_data(X):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = min_max_scaler.fit(X)
    return scaler.transform(X)


def cal_cv2(data):
    b = data.iloc[:, 1:].T.values
    c = b.copy()
    #
    means = np.mean(c, axis=1)
    variance = np.var(c, axis=1)
    cv2 = variance / means**2
    #
    minMeanForFit = np.quantile(means[np.where(cv2 > 0.5)], 0.95)
    useForFit = means >= minMeanForFit
    gamma_model = sm.GLM(cv2[useForFit],
                         np.array([np.repeat(1, means[useForFit].shape[0]), 1 / means[useForFit]]).T,
                         family=sm.families.Gamma(link=sm.genmod.families.links.identity))
    gamma_results = gamma_model.fit()
    a0 = gamma_results.params[0]
    a1 = gamma_results.params[1]
    afit = a1 / means + a0
    varFitRatio = variance / (afit * (means**2))
    return varFitRatio


def cal_pca(data):
    X = data.iloc[:, 1:]
    X_s = scale_data(X)
    pca = PCA(n_components=1)
    pca.fit(X_s)
    pc1_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return abs(pc1_loadings.T[0])


def feature_selection_rfc(data, label):
    X = data.iloc[:, 1:]
    X_s = scale_data(X)
    forest = RandomForestClassifier()
    forest.fit(X_s, label)
    importances = forest.feature_importances_
    return importances


def mic(data, label):
    X = data.iloc[:, 1:]
    X_s = scale_data(X)
    MIC = []
    m = MINE(alpha=0.6, c=15, est="mic_approx")
    for i in X_s.T:
        m.compute_score(i, label)
        MIC.append(m.mic())
    return MIC


def feature_selection_linear_svm(data, label):
    X = data.iloc[:, 1:]
    X_s = scale_data(X)
    clf = LinearSVC(multi_class="ovr", random_state=42)
    model = clf.fit(X_s, label)
    svm_weights = np.abs(model.coef_).sum(axis=0)
    svm_weights /= svm_weights.max()
    return svm_weights


def feature_selection_Turf(data, label, njobs):
    X = data.iloc[:, 1:]
    X_s = scale_data(X)
    head = list(X.columns)
    fs = TuRF(core_algorithm="ReliefF",
              n_features_to_select=2,
              pct=0.5,
              verbose=True,
              n_jobs=njobs)
    relif = fs.fit(X_s, label, head)
    return relif.feature_importances_


def cor_matrix(data, method):
    if method == "kendall":
        md = stats.kendalltau
    elif method == "spearman":
        return stats.spearmanr(data, axis=0)[0]
    else:
        md = stats.pearsonr
    data_shape = data.shape[1]
    out = np.zeros((data_shape, data_shape))
    site = 0
    for i in range(data_shape-1):
        a = data[:,i]
        out[i,i] = 1
        for j in range(i+1, data_shape):
            b = (data[:,j])
            person_cor, _ = md(a,b)
            out[i,j] = out[j,i] = person_cor
    out[data_shape-1, data_shape-1] = 1
    return out


# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/fselect/fselect.py
def fscore(data):
    X = data.iloc[:,1:].values
    y = data.iloc[:,0].values
    X_all_mean = np.mean(X,axis=0)
    f_bottom = np.repeat(1e-12, X.shape[1])
    f_top = np.zeros(X.shape[1])
    for i in np.unique(y):
        X_ = X[y==i]
        X_mean = np.mean(X_, axis=0)
        f_top += X_.shape[0]*np.square(X_mean-X_all_mean)
        f_bottom += np.sum(np.square(X_),axis=0)-(np.sum(X_,axis=0)**2)/X_.shape[0]
    return f_top/f_bottom