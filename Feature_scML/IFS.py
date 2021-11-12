import pandas as pd
import os
from .feature_selection import fs_output
from sklearn.model_selection import train_test_split
from .models import model_train


def overall_predict(args):
    filename = os.path.split(args.input_train)[1].split(".")[0]
    data = pd.read_csv(args.input_train, header=0)
    train_data = fs_output(data, args.method, filename, args.njobs)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data['Label'].values
    if args.output:
        os.chdir(args.output)
    if args.input_test:
        test_data = pd.read_csv(args.input_test, header=0)
        test_data.reindex(train_data.columns, axis=1)
        test_data.to_csv("test_{}.csv".format(args.input_test), index=None)
        X_test = test_data.iloc[:, 1:].values
        y_test = test_data['Label'].values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
    
    # get dataframe
    accuracy_data = pd.DataFrame(columns=["feature number", "train_accuracy",
                                          "test_accuracy", "parameters"])
    if not args.end:
        args.end = X_train.shape[1] + 1
    else:
        args.end += 1
    for i in range(args.start, args.end, args.step):
        X_train_ = X_train[:, :i]
        X_test_ = X_test[:, :i]
        filename_part = filename + "_" + str(X_train_.shape[1])
        train_accuracy, test_accuracy, best_params_ = model_train(X_train_, y_train, X_test_, y_test, 
                                                                  njob=args.njobs, 
                                                                  get_model=args.getmodel,
                                                                  method=args.classifier,
                                                                  filename=filename_part)
        feature_number = X_train_.shape[1]
        accuracy_data.loc[i] = feature_number, train_accuracy, test_accuracy, best_params_
<<<<<<< HEAD
    accuracy_data.to_csv("{0}-{1}_{2}_SVM_accuracy.csv".format(args.start, args.end - 1, args.method), index=False)
=======
    accuracy_data.to_csv("{0}-{1}_{2}_{3}_accuracy.csv".format(args.start, args.end - 1, args.method, args.classifier), index=False)
>>>>>>> 8bd37314557f03bbefbaf089a687d1810733c30f

