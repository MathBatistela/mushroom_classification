import numpy as np
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from joblib import Parallel, delayed
from tqdm.notebook import tqdm


def select_best_k_knn(ks, X_train, X_val, y_train, y_val, score_type='accuracy'):

    def train_knn(k, X_train, X_val, y_train, y_val):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_val)
        return {"accuracy_score": accuracy_score(y_val, pred), "f1_score": f1_score(y_val, pred)}

    scores_list = Parallel(n_jobs=4)(delayed(train_knn)(
        k, X_train, X_val, y_train, y_val) for k in ks)

    scores = None
    if score_type == 'accuracy':
        scores = [score['accuracy_score'] for score in scores_list]
    elif score_type == 'f1':
        scores = [score['f1_score'] for score in scores_list]
    else:
        raise ValueError('Undefined score type')

    best_score = max(scores)
    best_k = ks[np.argmax(scores)]
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(np.vstack((X_train, X_val)), [*y_train, *y_val])

    return knn, best_k, best_score


def do_cv_knn(X, y, cv_splits, ks, score_type='accuracy'):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    scores = []
    predicts = []

    pgb = tqdm(total=cv_splits, desc='Processed folds')

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, stratify=y_train, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)
        X_val = ss.transform(X_val)

        knn, _, _ = select_best_k_knn(
            ks, X_train, X_val, y_train, y_val, score_type)
        pred = knn.predict(X_test)

        if score_type == 'accuracy':
            scores.append(accuracy_score(y_test, pred))
        elif score_type == 'f1':
            scores.append(f1_score(y_test, pred))
        else:
            raise ValueError('Undefined score type')

        predicts.append((y_test, pred))

        pgb.update(1)

    pgb.close()

    return scores, predicts


def do_grid_cv_knn(X, y, cv_splits, ks, score_type='accuracy'):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    scores = []
    predicts = []

    pgb = tqdm(total=cv_splits, desc='Processed folds')

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)

        params = {'n_neighbors': ks}

        knn = KNeighborsClassifier()

        knn = GridSearchCV(knn, params, cv=StratifiedKFold(
            n_splits=cv_splits), scoring=score_type)

        knn.fit(X_train, y_train)

        pred = knn.predict(X_test)

        if score_type == 'accuracy':
            scores.append(accuracy_score(y_test, pred))
        elif score_type == 'f1':
            scores.append(f1_score(y_test, pred))
        else:
            raise ValueError('Undefined score type')

        predicts.append((y_test, pred))

        pgb.update(1)

    pgb.close()

    return scores, predicts


def select_best_svm(Cs, gammas, X_train: np.ndarray, X_val: np.ndarray,
                    y_train: np.ndarray, y_val: np.ndarray, n_jobs=4, score_type='accuracy'):

    def treinar_svm(C, gamma, X_train, X_val, y_train, y_val):
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_val)
        return {"accuracy_score": accuracy_score(y_val, pred), "f1_score": f1_score(y_val, pred)}

    combinacoes_parametros = list(itertools.product(Cs, gammas))

    scores_list = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                          (c, g, X_train, X_val, y_train, y_val) for c, g in combinacoes_parametros)

    scores = None
    if score_type == 'accuracy':
        scores = [score['accuracy_score'] for score in scores_list]
    elif score_type == 'f1':
        scores = [score['f1_score'] for score in scores_list]
    else:
        raise ValueError('Undefined score type')

    best_score = max(scores)
    best_comb = combinacoes_parametros[np.argmax(scores)]
    best_c = best_comb[0]
    best_gamma = best_comb[1]

    svm = SVC(C=best_c, gamma=best_gamma)
    svm.fit(np.vstack((X_train, X_val)), [*y_train, *y_val])

    return svm, best_comb, best_score


def do_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale'], score_type='accuracy'):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    scores = []
    predicts = []

    pgb = tqdm(total=cv_splits, desc='Processed folds')

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, stratify=y_train, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)
        X_val = ss.transform(X_val)

        svm, _, _ = select_best_svm(
            Cs, gammas, X_train, X_val, y_train, y_val, 4, score_type)
        pred = svm.predict(X_test)

        if score_type == 'accuracy':
            scores.append(accuracy_score(y_test, pred))
        elif score_type == 'f1':
            scores.append(f1_score(y_test, pred))
        else:
            raise ValueError('Undefined score type')

        predicts.append((y_test, pred))

        pgb.update(1)

    pgb.close()

    return scores, predicts


def do_grid_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale'], score_type='accuracy'):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    scores = []
    predicts = []
    c_params = []

    pgb = tqdm(total=cv_splits, desc='Processed folds')

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)

        params = {'C': Cs, 'gamma': gammas}
        svm = SVC()
        svm = GridSearchCV(svm, params, cv=StratifiedKFold(
            n_splits=cv_splits), scoring=score_type)
        svm.fit(X_train, y_train)

        pred = svm.predict(X_test)

        if score_type == 'accuracy':
            scores.append(accuracy_score(y_test, pred))
        elif score_type == 'f1':
            scores.append(f1_score(y_test, pred))
        else:
            raise ValueError('Undefined score type')

        predicts.append((y_test, pred))
        c_params.append(svm.best_params_)

        pgb.update(1)

    pgb.close()

    return scores, predicts, c_params
