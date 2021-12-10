import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

trainData = pd.read_csv(
    'trainingData/training_data_classification.csv')

testData = pd.read_csv(
    'testingData/testing_data_classification.csv')

answerData = pd.read_csv('testingData/answer_data_classification.csv')


# (x_train, x_test, y_train, y_test) = train_test_split(
#     trainData.iloc[:, 0:5].values, trainData.iloc[:, 5].values, test_size=1)

x_train = trainData.iloc[:, 0:5].values
x_test = testData.iloc[:, 0:5].values
y_train = trainData.iloc[:, 5].values
y_test = answerData.iloc[:, 5].values


# df = pd.read_csv('testingData/heart.csv')

# (x_train, x_test, y_train, y_test) = train_test_split(
#     df.iloc[:, 0:13].values, df.iloc[:, 13].values, train_size=0.8)


class MLClassifier:
    def removeOutliers(self, x: np.ndarray, y: np.ndarray) -> None:
        covariance = np.cov(x, rowvar=False)
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)
        centerpoint = np.mean(x, axis=0)

        distances = []

        for i, val in enumerate(x):
            p1 = val
            p2 = centerpoint
            distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
            distances.append(distance)

        cutoff = chi2.ppf(.5, x.shape[1])

        outlierIndexes = np.where(distances > cutoff)

        x = np.delete(x, outlierIndexes[0], axis=0)
        y = np.delete(y, outlierIndexes[0])
        return (x, y)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        '''

        # Removed outlier detection and removal since there are no significant outlying data elements.
        # (x, y) = self.removeOutliers(x, y)

        self.d = x.shape[1]

        # no. of classes; assumes labels to be integers from 0 to nclasses-1
        self.nclasses = len(set(y))

        # list of means; mu_list[i] is mean vector for label i
        self.mu_list = []

        # list of inverse covariance matrices;
        # sigma_list[i] is inverse covariance matrix for label i
        # for efficiency reasons we store only the inverses
        self.sigma_inv_list = []

        # list of scalars in front of e^...
        self.scalars = []

        n = x.shape[0]
        for i in range(self.nclasses):

            # subset of obesrvations for label i
            cls_x = np.array([x[j] for j in range(n) if y[j] == i+1])

            mu = np.mean(cls_x, axis=0)

            # rowvar = False, this is to use columns as variables instead of rows
            sigma = np.cov(cls_x, rowvar=False)
            if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:
                # if at least one eigenvalue is <= 0 show warning
                print(
                    f'Warning! Covariance matrix for label {cls_x} is not positive definite!\n')

            sigma_inv = np.linalg.inv(sigma)

            scalar = 1/np.sqrt(((2*np.pi)**self.d)*np.linalg.det(sigma))

            self.mu_list.append(mu)
            self.sigma_inv_list.append(sigma_inv)
            self.scalars.append(scalar)

    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        '''
        x - numpy array of shape (d,)
        cls - class label

        Returns: likelihood of x under the assumption that class label is cls
        '''
        mu = self.mu_list[cls]
        sigma_inv = self.sigma_inv_list[cls]
        scalar = self.scalars[cls]
        # d = self.d
        exp = (-1/2)*np.dot(np.matmul(x-mu, sigma_inv), x-mu)

        return scalar * (np.e**exp)

    def predict(self, x: np.ndarray) -> int:
        '''
        x - numpy array of shape (d,)
        Returns: predicted label
        '''
        likelihoods = [self._class_likelihood(
            x, i) for i in range(self.nclasses)]
        return np.argmax(likelihoods)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        Returns: accuracy of predictions
        '''
        n = x.shape[0]
        predicted_y = np.array([self.predict(x[i])+1 for i in range(n)])
        n_correct = np.sum(predicted_y == y)
        return n_correct/n


mlc = MLClassifier()

mlc.fit(x_train, y_train)

n = x_test.shape[0]

print("Predictions: ", np.array(
    [mlc.predict(x_test[i])+1 for i in range(n)]))

print("The score is: ", mlc.score(x_test, y_test))
