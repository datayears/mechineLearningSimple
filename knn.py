import numpy as np
import operator
from numpy import array

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    print 'dataSetSize=',dataSetSize
    print 'diffMat=',diffMat
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    print 'sqDiffMat=',sqDiffMat
    print 'sqDistances=',sqDistances
    distances=sqDistances**0.5
    print 'distances=',distances
    sortedDistIndicies=distances.argsort()
    print 'sortedDistIndicies=',sortedDistIndicies
    classCount={}
    for i in range(k):
        votellabel = labels[sortedDistIndicies[i]]
        print '---------------------'
        print 'i=',i,'votellabel=',votellabel
        classCount[votellabel]=classCount.get(votellabel,0)+1
        #print 'classCount',classCount
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    print 'sortedClassCount=',sortedClassCount
    return sortedClassCount[0][0]

if __name__ == "__main__":
    group,labels=createDataSet()
    s=classify0([0,0.2],group,labels,3)
    print s
