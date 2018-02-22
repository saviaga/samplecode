import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import pandas
import numpy as np
from scipy import interp
## SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn import  metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class MeanEmbeddingVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.items())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.items())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def load_data_and_labels(data_path):
    data_file = pandas.read_csv(data_path)
    x_text = data_file.Text.tolist()
    x_text = [s.strip() for s in x_text]
    labels = data_file.Class.tolist()
    return x_text,labels

def load_data(data_path):
    data_file = pandas.read_csv(data_path)
    x_text = data_file.Text.tolist()
    return [s.strip() for s in x_text], [s.strip().split() for s in x_text]

def preprocess_w2v(X, y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        X_train_value, X_test_value = X[train_index], X[test_index]
        y_train_value, y_test_value = y[train_index], y[test_index]
        X_train.append(X_train_value)
        X_test.append(X_test_value)
        y_train.append(y_train_value)
        y_test.append(y_test_value)
    return X_train, X_test, y_train, y_test

def preprocess_bag(X, y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        features_train, features_test = X[train_index], X[test_index]
        t_labels_train, t_labels_test = y[train_index], y[test_index]
        vectorizer = TfidfVectorizer(stop_words='english')
        features_train = vectorizer.fit_transform(features_train)
        features_test  = vectorizer.transform(features_test)
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(features_train, t_labels_train)
        X_train.append(selector.transform(features_train).toarray())
        X_test.append(selector.transform(features_test).toarray())
        y_train.append(t_labels_train)
        y_test.append(t_labels_test)
    return X_train, X_test, y_train, y_test

def train(clf, X_train, X_test, y_train, y_test):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    acc_tot = precision_tot = recall_tot = f1_tot = support_tot = 0
    headers = 'fl,precision,recall,f1,support,acc,TN, FP, FN, TP'
    print(headers)
    for fl in range(FOLDS):
        clf.fit(X_train[fl], y_train[fl])
        pred = clf.predict(X_test[fl])
        acc = metrics.accuracy_score(y_test[fl], pred)
        conf_matrix = metrics.confusion_matrix(y_test[fl], pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_test[fl], pred, average="weighted")
        support = conf_matrix[0][0]+conf_matrix[0][1]+conf_matrix[1][0]+conf_matrix[1][1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test[fl], pred)
        roc_auc = metrics.auc(fpr, tpr)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (fl, precision, recall, f1, support, acc, conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]))
        acc_tot += acc
        precision_tot += precision
        recall_tot += recall
        f1_tot += f1
        support_tot += support
    mean_tpr /= FOLDS
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    acc_tot /= FOLDS
    precision_tot /= FOLDS
    recall_tot /= FOLDS
    f1_tot /= FOLDS
    support_tot /= FOLDS
    print("Total accuracy: "+str(acc_tot))
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_auc': mean_auc
    }

def plot(title, records):
    lw = 2
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    for record in records:
        desc = '%s (area = %0.2f)' % (record['label'], record['mean_auc'])
        plt.plot(record['mean_fpr'], record['mean_tpr'], linestyle='--', label=desc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def benchmark(params):
    # load dataset
    url = "train_morphine.csv"
    names = ['Class', 'Text']
    dataframe = pandas.read_csv(url, names=names, header=1)
    array = dataframe.values
    X = array[:, 0:1]
    Y = array[:, 1]
    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=FOLDS, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def predict(params):
    clf = ExtraTreesClassifier(n_estimators=200)
    with open(params["w2v_file"], "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    clf_w2v = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("classifier", clf)])
    X, y = load_data_and_labels(params["train_file"])
    col_y, arr_X, arr_y = {}, [], []
    for i in range(len(y)):
        if y[i] not in col_y:
            col_y[y[i]] = []
        col_y[y[i]].append(X[i])
    for field in col_y:
        arr_X.append(col_y[field])
        arr_y.append(field)
    clf_w2v.fit(arr_X, arr_y)
    raw_data = {}
    raw_data['Text'], tokenized = load_data(params["input_file"])
    raw_data['Class'] = clf_w2v.predict(tokenized)
    data_frame = pandas.DataFrame(raw_data)
    data_frame.to_csv(params["output_file"])
    print(data_frame)
    return raw_data

def run(params):
    if params["bench"]:
        benchmark(params)
    else:
        predict(params)



if __name__ == "__main__":

  FOLDS = 10
  params = {
      "bench": True,
      "train_file": "train.csv",
      "input_file": "predict.csv",
      "output_file": "results.csv",
      "w2v_file": "glove.twitter.27B.100d.txt"
  }
  run(params)