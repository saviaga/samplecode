import math
import pickle


vectorsAuthorsArray = pickle.load( open( "vectorsAuthorsArray.p", "rb" ) )
print (len(vectorsAuthorsArray))
	
NUM_CLUSTERS = 3
TOTAL_DATA = len(vectorsAuthorsArray)
LOWEST_SAMPLE_POINT = 0 #element 0 of SAMPLES.
HIGHEST_SAMPLE_POINT = 3 #element 3 of SAMPLES.
MEDIAN_SAMPLE_POINT = 6 #element 3 of SAMPLES.
FOURTH_SAMPLE_POINT = 10 #element 3 of SAMPLES.

BIG_NUMBER = math.pow(10, 10)

print (vectorsAuthorsArray)
SAMPLES=vectorsAuthorsArray
print (SAMPLES)
data = []
centroids = []
class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def set_x(self, x):
        self.x = x
    
    def get_x(self):
        return self.x
    
    def set_y(self, y):
        self.y = y
    
    def get_y(self):
        return self.y
    
    def set_cluster(self, clusterNumber):
        self.clusterNumber = clusterNumber
    
    def get_cluster(self):
        return self.clusterNumber

class Centroid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def set_x(self, x):
        self.x = x
    
    def get_x(self):
        return self.x
    
    def set_y(self, y):
        self.y = y
    
    def get_y(self):
        return self.y

def initialize_centroids():
    # Set the centoid coordinates to match the data points furthest from each other.
    # In this example, (1.0, 1.0) and (5.0, 7.0)
    centroids.append(Centroid(SAMPLES[LOWEST_SAMPLE_POINT][0], SAMPLES[LOWEST_SAMPLE_POINT][1]))
    centroids.append(Centroid(SAMPLES[HIGHEST_SAMPLE_POINT][0], SAMPLES[HIGHEST_SAMPLE_POINT][1]))
    centroids.append(Centroid(SAMPLES[MEDIAN_SAMPLE_POINT][0], SAMPLES[MEDIAN_SAMPLE_POINT][1]))
    centroids.append(Centroid(SAMPLES[FOURTH_SAMPLE_POINT][0], SAMPLES[FOURTH_SAMPLE_POINT][1]))
    
    #FOURTH_SAMPLE_POINT

    #MEDIAN_SAMPLE_POINT 
    print("Centroids initialized at:")
    print("(", centroids[0].get_x(), ", ", centroids[0].get_y(), ")")
    print("(", centroids[1].get_x(), ", ", centroids[1].get_y(), ")")
    print("(", centroids[0].get_x(), ", ", centroids[1].get_y(), ")")
    print("(", centroids[1].get_x(), ", ", centroids[0].get_y(), ")")
    
    #print()
    return

def initialize_datapoints():
    # DataPoint objects' x and y values are taken from the SAMPLE array.
    # The DataPoints associated with LOWEST_SAMPLE_POINT and HIGHEST_SAMPLE_POINT are initially
    # assigned to the clusters matching the LOWEST_SAMPLE_POINT and HIGHEST_SAMPLE_POINT centroids.
    for i in range(TOTAL_DATA):
        newPoint = DataPoint(SAMPLES[i][0], SAMPLES[i][1])
        
        if(i == LOWEST_SAMPLE_POINT):
            newPoint.set_cluster(0)
        elif(i == HIGHEST_SAMPLE_POINT):
            newPoint.set_cluster(1)
        elif(i==MEDIAN_SAMPLE_POINT):
        	newPoint.set_cluster(2)
        elif(i==FOURTH_SAMPLE_POINT):
        	newPoint.set_cluster(3)
        #FOURTH_SAMPLE_POINT 
        else:
            newPoint.set_cluster(None)
            
        data.append(newPoint)
    
    return

def get_distance(dataPointX, dataPointY, centroidX, centroidY):
    # Calculate Euclidean distance.
    return math.sqrt(math.pow((centroidY - dataPointY), 2) + math.pow((centroidX - dataPointX), 2))

def recalculate_centroids():
    totalX = 0
    totalY = 0
    totalInCluster = 0
    
    for j in range(NUM_CLUSTERS):
        for k in range(len(data)):
            if(data[k].get_cluster() == j):
                totalX += data[k].get_x()
                totalY += data[k].get_y()
                totalInCluster += 1
        
        if(totalInCluster > 0):
            print ("TOTAL X:"+str(totalX))
            print ("TOTAL totalInCluster:"+str(totalInCluster))
            i=int(totalX / totalInCluster)
            print ("FLOATING:"+str(i))
            print ("len centroid"+str(len(centroids)))
            print ("len centroid"+str(len(centroids)))
            centroids[j].set_y(totalY / totalInCluster)
            centroids[j].set_x(totalX / totalInCluster)
            centroids[j].set_y(totalY / totalInCluster)
    
    return

def update_clusters():
    isStillMoving = 0
    
    for i in range(TOTAL_DATA):
        bestMinimum = BIG_NUMBER
        currentCluster = 0
        
        for j in range(NUM_CLUSTERS):
            distance = get_distance(data[i].get_x(), data[i].get_y(), centroids[j].get_x(), centroids[j].get_y())
            if(distance < bestMinimum):
                bestMinimum = distance
                currentCluster = j
        
        data[i].set_cluster(currentCluster)
        
        if(data[i].get_cluster() is None or data[i].get_cluster() != currentCluster):
            data[i].set_cluster(currentCluster)
            isStillMoving = 1
    
    return isStillMoving

def perform_kmeans():
    isStillMoving = 1
    initialize_centroids()
    initialize_datapoints()
    
    while(isStillMoving):
        recalculate_centroids()
        isStillMoving = update_clusters()
    
    return

def print_results():
    clustersStored={}
	for i in range(NUM_CLUSTERS):
		print("Cluster ", i, " includes:")
		clustersStored.setdefault(i,{})
		for j in range(TOTAL_DATA):
			if(data[j].get_cluster() == i):
				print("(", data[j].get_x(), ", ", data[j].get_y(), ")")
				x=data[j].get_x()
				y=data[j].get_y()
				keyAuthor=str(x)+","+str(y)
				clustersStored[i].setdefault(keyAuthor,0)
				clustersStored[i][keyAuthor]+=1

		print()
	pickle.dump(clustersStored, open( "clustersStored.p", "wb" ))



if __name__=="__main__":
    perform_kmeans()
    print_results()
