#INFO3401-Kexin Zhai
import pandas as pd
import seaborn as sns
#Import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def loadData(datafile):
    with open(datafile, 'r', encoding = "latin1") as csvfile:
        data = pd.read_csv(csvfile)
    
    # Inspect the data
    print(data.columns.values)
    
    return data

def runKNN(dataset, prediction, ignore, n_neighbors):
    #Set up our dataset
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values
    
    # Split the data into a training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1, stratify = Y)#original: test_size = 0.2
    
    # Run k-NN algorithm
    knn = KNeighborsClassifier(n_neighbors)
    
    # Train the model
    knn.fit(X_train, Y_train)
    
    Y_pred = knn.predict(X_test)
    # Test the model
    
    score = knn.score(X_test, Y_test)
    f1 = f1_score(Y_test, Y_pred, average='macro')#y_true, y_pred
    print("Predicts " + prediction + " with " + str(score) + " accuracy")
    print("f1_score: "+ str(f1))
    print("Chance is " + str(1.0/len(dataset.groupby(prediction))))
    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns=[prediction, ignore])
    
    # Determine the five closest neighbors
    neighbors = model.kneighbors(X, n_neighbors = 5, return_distance=False)
    
    # Print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])
        
def runKNNCrossfold(dataset, k_values):
    X = dataset.drop(columns=["pos", "player"])
    Y = dataset["pos"].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1, stratify = Y)
    cv_scores = []
    knn = KNeighborsClassifier(n_neighbors=5)
    for k in k_values:
        scores = cross_val_score(knn, X_train, Y_train, cv=k, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(str(k)+"-fold cross validation with " + str(scores) + " accuracy")
    return cv_scores
    
def determineK(dataset):
    # determining best k
    optimal_k_index = dataset.index(max(dataset))
    optimal_k_accuracy = dataset[optimal_k_index]
    print ("The resulting accuracy of optimal k is " + str(optimal_k_accuracy))
    return optimal_k_index

def runKMeans(dataset, ignore, n_clusters):
    # Set up out dataset
    X = dataset.drop(columns=ignore)
    
    # Run k-Means algorithm
    kmeans = KMeans(n_clusters)
    
    # Train the model
    kmeans.fit(X)
    
    # Add the predictions to the dataframe
    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)
    
    # Print a scatterplot matrix
    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette = 'Set2')
    
    scatterMatrix.savefig("kmeanClusters.png")
    
    return kmeans

def findClusterK(dataset, ignore):
    X = dataset.drop(columns=ignore)
    Sum_of_squared_distances = []
    K = range(1,8)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# 
# Test your code
#MON.1/WED.2
nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player", 5)
print("classifyplayer:")
classifyPlayer(nbaData.loc[nbaData['player'] == 'LeBron James'], nbaData, knnModel, 'pos', 'player')
#Predicts pos with 0.4816753926701571 accuracy
#f1_score: 0.4845543071161048
#Chance is 0.2
#
#WED.3
#Both accuracy and f1 score close to 48% which means this model is not a good classifyer to predict a player's position. Because the accuracy is almost about 50%.
#WED.4
k_values = [5,7,10]
cv_scores = runKNNCrossfold(nbaData, k_values)
print(cv_scores)
#5-fold cross validation with [0.33898305 0.32758621 0.47368421 0.41071429 0.45454545] accuracy
#7-fold cross validation with [0.36363636 0.36363636 0.31707317 0.43589744 0.38461538 0.46153846
# 0.41025641] accuracy
#10-fold cross validation with [0.38709677 0.26666667 0.37931034 0.27586207 0.4137931  0.5
# 0.39285714 0.46428571 0.51851852 0.38461538] accuracy
#The mean accuracy of each fold cross validation is
#[0.40110264170601306, 0.3909505129017324, 0.39830057183783546]
#WED.5
index = determineK(cv_scores)
print("And the optimal k is " + str(k_values[index]))
#The resulting accuracy of optimal k is 0.40110264170601306
#And the optimal k is 5
#FRI.6
kmeansModel = runKMeans(nbaData, ['pos','player'], 5)
#FRI.7
findClusterK(nbaData, ['pos','player'])
#In the plot the function created the elbow is at k=3 indicating the optimal k for this dataset is 3