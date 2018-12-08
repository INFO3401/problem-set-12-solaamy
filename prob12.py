import pandas as pd
import seaborn as sns
#Import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
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
    knn = KNeighborsClassifier(n_neighbors=5)
    
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

def runKMeans(dataset, ignore):
    # Set up out dataset
    X = dataset.drop(columns=ignore)
    
    # Run k-Means algorithm
    kmeans = KMeans(n_clusters=5)
    
    # Train the model
    kmeans.fit(X)
    
    # Add the predictions to the dataframe
    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)
    
#    # Print a scatterplot matrix
#    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette = 'Set2')
#    
#    scatterMatrix.savefig("kmeanClusters.png")
    
    return kmeans

# 
# Test your code
nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player", 5)
print("classifyplayer:")
classifyPlayer(nbaData.loc[nbaData['player'] == 'LeBron James'], nbaData, knnModel, 'pos', 'player')

kmeansModel = runKMeans(nbaData, ['pos','player'])