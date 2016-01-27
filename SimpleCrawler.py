from bs4 import BeautifulSoup
import urllib.request
import NodeRecognizer
import json
from pprint import pprint
import DataPrepare
import numpy



def GetDNNInputFromWebsite(url):
    f = urllib.request.urlopen(url)
    content = f.read()
    soup = BeautifulSoup(content, "lxml")

    NodeRecognizer.NodeRecognizer(soup.body, 0)

    featureList = NodeRecognizer.featureList

    dataP = DataPrepare.DataPrepare()
    dataP.GetDNNInput(featureList)
    return dataP.labels,numpy.stack(dataP.DNNSamples)

def GetTrainData():
    trainSet = GetDNNInputFromWebsite("http://www.cnbeta.com/topics/9.htm")
    testSet = GetDNNInputFromWebsite("http://www.cnbeta.com/topics/8.htm")
    return [trainSet, testSet]
