import numpy as np

'''
This code follows the algorithm for ReliefF as described in 
"An adaptation of Relief for attribute estimation in regression"
by M. Robnik-Sikonja and I. Kononenko

Equation References in comments are based on the aforementioned article

To work with RReliefF, use RReliefF(X, y, opt) 
opt can be replaced with the following optional arguments

- updates - This can be 'all' (default) or a positive integer depending 
- k - The number of neighbours to look at. Default is 10.
- sigma - Distance scaling factor. Default is 50.
- weight_track - Returns a matrix which tracks the weight changes at each iteration. False by default
- categoricalx - This aspect has not been properly assimilated yet. Future work. Intended function:
You can specify if your inputs are categorial or not (False by default - assumes inputs are numeric).
Does not allow for the mixing of numeric and categorical predictors
'''


''' Multiple KNN Search functions for the different algorithms'''

# This function finds the k nearest neighbours
def __knnsearchR(A, b, n):

    difference = (A - b)**2
    sumDifference = np.sum(difference, axis = 1)
    neighbourIndex = np.argsort(sumDifference)
    neighbours = A[neighbourIndex][1:]
    knn = neighbours[:n]
    return knn, neighbourIndex[1:] #Don't want to count the original point

# This function finds the k nearest neighbours
def __knnsearchF(A, b, n, y, label):

    indToKeep = y==label

    A = A[indToKeep.ravel(), :]


    difference = (A - b)**2
    sumDifference = np.sum(difference, axis = 1)
    neighbourIndex = np.argsort(sumDifference)
    neighbours = A[neighbourIndex][1:]
    knn = neighbours[:n]
    return knn, neighbourIndex[1:] #Don't want to count the original point

# This function finds the k nearest neighbours
def __knnsearch(A, b, n, opt, y, yRandomInstance):

    if opt == 'hit':
        indToKeep = y == yRandomInstance
    else:
        indToKeep = y!=yRandomInstance

    A = A[indToKeep, :]


    difference = (A - b)**2
    sumDifference = np.sum(difference, axis = 1)
    neighbourIndex = np.argsort(sumDifference)
    neighbours = A[neighbourIndex][1:]
    knn = neighbours[:n]
    return knn, neighbourIndex[1:] #Don't want to count the original point



'''------------> Helper Functions <------------'''


# This follows the Eqn 8
def __distance(k, sigma):
    d1 = [np.exp(-((n + 1) / sigma) ** 2) for n in range(k)]
    d = d1 / np.sum(d1)
    return d


# This follows Eqn 2
def __diffNumeric(A, XRandomInstance, XKNNj, X):
    denominator = np.max(X[:, A]) - np.min(X[:, A])

    return np.abs(XRandomInstance[A] - XKNNj[A]) / denominator


def __diffCaterogical(A, XRandomInstance, XKNNj, X):
    return int(not XRandomInstance[A] == XKNNj[A])

def __probability_class(y, currentLabel):

    numCurrentLabel = np.sum(y == currentLabel)
    numTotal = len(y)
    return numCurrentLabel/numTotal

'''------------> Main Relief related functions <------------'''

def RReliefF(X, y, updates='all', k=10, sigma=30, weight_track=False, categoricalx = False):

    # Check if user wants all values to be considered
    if updates == 'all':
        m = X.shape[0]
    else:
        m = updates

    # The constants need for RReliefF
    N_dC = 0
    N_dA = np.zeros([X.shape[1],1])
    N_dCanddA= np.zeros([X.shape[1],1])
    W_A = np.zeros([X.shape[1],1])
    Wtrack = np.zeros([m, X.shape[1]])
    yRange = np.max(y) - np.min(y)
    iTrack = np.zeros([m,1])


    # Check if the input is categorical
    if categoricalx:
        __diff = __diffCaterogical
    else:
        __diff = __diffNumeric

    # Repeat based on the total number of inputs or based on a user specified value
    for i in range(m):

        # Randomly access an instance
        if updates == 'all':
            random_instance = i
        else:
            random_instance = np.random.randint(low=0, high=X.shape[0])

        # Select a 'k' number in instances near the chosen random instance
        XKNN, neighbourIndex = __knnsearchR(X, X[random_instance,:],k)
        yKNN = y[neighbourIndex]
        XRandomInstance = X[random_instance, :]
        yRandomInstance = y[random_instance]

        # Loop through all selected random instances
        for j in range(k):

            # Weight for different predictions
            N_dC += (np.abs(yRandomInstance-yKNN[j])/yRange) * __distance(k, sigma)[j]

            # Loop through all attributes
            for A in range(X.shape[1]):

                # Weight to account for different attributes
                N_dA[A] = N_dA[A] +  __diff(A, XRandomInstance, XKNN[j], X) * __distance(k, 30)[j]

                # Concurrent examination of attributes and output
                N_dCanddA[A] = N_dCanddA[A] + (np.abs(yRandomInstance-yKNN[j])/yRange) * __distance(k, sigma)[j] *\
                               __diff(A, XRandomInstance, XKNN[j], X)

        # This is another variable we use to keep track of all weights - this can be used to see how RReliefF works
        for A in range(X.shape[1]):
            Wtrack[i, A] = N_dCanddA[A] / N_dC - ((N_dA[A] - N_dCanddA[A]) / (m - N_dC))

        # The index corresponding to the weight
        iTrack[i] = random_instance

    # Calculating the weights for all features
    for A in range(X.shape[1]):
        W_A[A] = N_dCanddA[A]/N_dC - ((N_dA[A]-N_dCanddA[A])/(m-N_dC))

    # Check if weight tracking is on
    if not weight_track:
        return W_A
    else:
        return W_A, Wtrack, iTrack


def ReliefF(X, y, updates='all', k=10, sigma=30, weight_track=False, categoricalx=False):
    # Check if user wants all values to be considered
    if updates == 'all':
        m = X.shape[0]
    else:
        m = updates

    # The constants need for RReliefF

    W_A = np.zeros([X.shape[1], 1])
    Wtrack = np.zeros([m, X.shape[1]])
    iTrack = np.zeros([m,1])
    # yRange = np.max(y) - np.min(y)

    # Find unique labels
    labels = np.unique(y)

    # Check if the input is categorical
    if categoricalx:
        __diff = __diffCaterogical
    else:
        __diff = __diffNumeric

    # Repeat based on the total number of inputs or based on a user specified value
    for i in range(m):

        # Randomly access an instance
        if updates == 'all':
            random_instance = i
        else:
            random_instance = np.random.randint(low=0, high=X.shape[0])


        iTrack[i] = random_instance
        currentLabel = y[random_instance]
        XKNNHit, neighbourIndexHit = __knnsearchF(X, X[random_instance, :], k, y, currentLabel)
        missedLabels = labels[labels != currentLabel]
        XKNNMiss = []
        neighbourIndexMiss = []

        # Go through and find the misses
        for n in range(len(missedLabels)):
            XKNNCurrentMiss, neighbourIndexCurrentMiss = __knnsearchF(X, X[random_instance, :], k, y, missedLabels[n])
            XKNNMiss.append(XKNNCurrentMiss)
            neighbourIndexMiss.append(neighbourIndexCurrentMiss)
        amrit = 2


        XRandomInstance = X[random_instance, :]
        yRandomInstance = y[random_instance]

        # Loop through all attributes
        for A in range(X.shape[1]):

            diffHit = 0
            # Loop through all neighbours
            for j in range(k):
                diffHit += __diff(A, XRandomInstance, XKNNHit[j], X)
            diffHit /= m * k

            diffMiss = 0
            # Loop through the missed labels
            for n in range(len(missedLabels)):

                diffCurrentMiss = 0

                # Loop through the neighbours
                for j in range(k):
                    diffCurrentMiss += __diff(A, XRandomInstance, XKNNMiss[n][j], X)

                diffMiss += __probability_class(y, missedLabels[n]) * diffCurrentMiss / (m * k)

            # Calculate the weight
            W_A[A] = W_A[A] - diffHit + diffMiss

            # Track the weights
            Wtrack[i, A] = W_A[A]

    # Check if weight tracking is on
    if not weight_track:
        return W_A
    else:
        return W_A, Wtrack, iTrack


def Relief(X, y, updates='all', sigma=30, weight_track=False, categoricalx=False):

    # Check if user wants all values to be considered
    if updates == 'all':
        m = X.shape[0]
    else:
        m = updates

    # The constants need for RReliefF

    W_A = np.zeros([X.shape[1], 1])
    Wtrack = np.zeros([m, X.shape[1]])
    hitTrack = np.zeros([m, X.shape[1]])
    missTrack = np.zeros([m, X.shape[1]])
    iTrack = np.zeros([m,1])
    # yRange = np.max(y) - np.min(y)

    # Check if the input is categorical
    if categoricalx:
        __diff = __diffCaterogical
    else:
        __diff = __diffNumeric

    # Repeat based on the total number of inputs or based on a user specified value
    for i in range(m):

        # Randomly access an instance
        if updates == 'all':
            random_instance = i
        else:
            random_instance = np.random.randint(low=0, high=X.shape[0])

        # Select a 'k' number in instances near the chosen random instance
        XKNNHit, neighbourIndexHit = __knnsearch(X, X[random_instance, :], 1, 'hit', y, y[random_instance])
        yKNNHit = y[neighbourIndexHit]
        XKNNMiss, neighbourIndexMiss = __knnsearch(X, X[random_instance, :], 1, 'miss', y, y[random_instance])
        yKNNMiss = y[neighbourIndexMiss]


        XRandomInstance = X[random_instance, :]
        yRandomInstance = y[random_instance]

        iTrack[i] = random_instance

        # Loop through all attributes
        for A in range(X.shape[1]):
            # Calculate the weight
            W_A[A] = W_A[A] - __diff(A, XRandomInstance, XKNNHit[0], X) / m + __diff(A, XRandomInstance, XKNNMiss[0],
                                                                                     X) / m
            # Track the weights
            Wtrack[i, A] = W_A[A]
            hitTrack[i, A] = __diff(A, XRandomInstance, XKNNHit[0], X) / m
            missTrack[i, A] = __diff(A, XRandomInstance, XKNNMiss[0], X) / m

    # Check if weight tracking is on
    if not weight_track:
        return W_A
    else:
        return W_A, Wtrack, iTrack, hitTrack, missTrack

