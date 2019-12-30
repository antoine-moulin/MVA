import sys
from lda import LDA
from log_reg import LogisticRegression
from lin_reg import LinearRegression
from qda import QDA
from utils import plot_results, read_data

if __name__ == '__main__':
    # read arguments
    num_args = len(sys.argv)
    if num_args == 1:
        params = {'idx_dataset': 'A'}
    elif num_args == 2:
        params = {'idx_dataset': sys.argv[1]}
    elif num_args == 3:
        params = {'idx_dataset': sys.argv[1],
                  'train': sys.argv[2] == 'True'}
    elif num_args == 4:
        params = {'idx_dataset': sys.argv[1],
                  'train': sys.argv[2] == 'True',
                  'test': sys.argv[3] == 'True'}

    # load dataset
    X_train, y_train, X_test, y_test = read_data(**params)

    if X_train is not None:
        # Linear Discriminant Analysis (LDA)
        lda = LDA()
        lda.fit(X_train, y_train)
        plot_results(lda, params['idx_dataset'], X_train, y_train, X_test, y_test)
        print('The accuracy on train (test) dataset {} for LDA: {} ({})'.format(params['idx_dataset'],
                                                                                lda.score(X_train, y_train),
                                                                                lda.score(X_test, y_test)))

        # Logistic regression
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        plot_results(log_reg, params['idx_dataset'], X_train, y_train, X_test, y_test)
        print('The accuracy on train (test) dataset {} for LogReg: {} ({})'.format(params['idx_dataset'],
                                                                                   log_reg.score(X_train, y_train),
                                                                                   log_reg.score(X_test, y_test)))

        # Linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        plot_results(lin_reg, params['idx_dataset'], X_train, y_train, X_test, y_test)
        print('The accuracy on train (test) dataset {} for LinReg: {} ({})'.format(params['idx_dataset'],
                                                                                   lin_reg.score(X_train, y_train),
                                                                                   lin_reg.score(X_test, y_test)))

        # Quadratic Discriminant Analysis (QDA)
        qda = QDA()
        qda.fit(X_train, y_train)
        plot_results(qda, params['idx_dataset'], X_train, y_train, X_test, y_test)
        print('The accuracy on train (test) dataset {} for QDA: {} ({})'.format(params['idx_dataset'],
                                                                                qda.score(X_train, y_train),
                                                                                qda.score(X_test, y_test)))

    else:
        plot_results(None, params['idx_dataset'], X_train, y_train, X_test, y_test)
