from pprint import pprint
import bs4.element
from bs4 import BeautifulSoup
import pprint
import numpy


class DataPrepare:

    def __init__ (self,
                  maxDepth = 5,
                  maxPositiveSample = 40,
                  maxNegtiveSample = 40):
        self.wordDict = {}
        self.wordCount = 0
        self.maxChildCount = 5
        self.maxDepth = maxDepth
        self.maxWordCount = 20
        self.DNNSamples = []
        self.labelsList = []
        # for debug
        self.maxPositiveSample = maxPositiveSample
        self.maxNegtiveSample = maxNegtiveSample


    def FindStrIndex(self, str):
        if str in self.wordDict:
            self.wordDict[str]['count'] +=1
            return self.wordDict[str]['index']
        else:
            self.wordDict[str]= {
                'count' : 1,
                'index' : self.wordCount
            }
            self.wordCount += 1
            return self.wordCount -1

    def ConvStr2Index(self, node):
        index = []
        if type(node) == str:
            return self.FindStrIndex(node)
        if type(node) ==  list :
            for child in node:
                temp_list = self.ConvStr2Index(child)
                index.append(temp_list)
        return index

    def ConvIndex2Vector(self, node, depth, left, right):
        selfInfo = node[0]
        for str in selfInfo:
            if str in self.sortedWordDict:
                index =  self.sortedWordDict[str]
                self.DNNSample[left][index] +=1

        childsInfo = node[1:]
        limitedChild = min(len(childsInfo),self.maxChildCount)
        for i in range(0, limitedChild):
            if depth < self.maxDepth:
                newLeft = left + i * self.maxChildCount ** (self.maxDepth - depth - 2)
                newRight = left + (i + 1) * self.maxChildCount ** (self.maxDepth - depth - 2)
            else:
                newLeft = left
                newRight = right
            self.ConvIndex2Vector(childsInfo[i], depth + 1, newLeft, newRight)

    def GetDNNInput(self, featuresListStr):
        # round 1```
        featuresListIndex = []
        for sample in featuresListStr:
            #print(sample)
            featuresListIndex.append(self.ConvStr2Index(sample['features']))


        sortedWordList = sorted(self.wordDict.items(),  key=lambda d:d[1]['count'], reverse=True)
        self.sortedWordDict = {}
        for i in range(0,self.maxWordCount):
            key = sortedWordList[i][0]
            self.sortedWordDict[key] = i

        # round 2
        positiveSampleCount = 0
        negtiveSampleCount = 0
        for sample in featuresListStr:
            flag = False
            if sample['label'] == 0 and negtiveSampleCount < self.maxNegtiveSample:
                flag = True
                negtiveSampleCount +=1
            if sample['label'] == 1 and positiveSampleCount < self.maxPositiveSample :
                flag = True
                positiveSampleCount +=1

            if flag:
                self.labelsList.append(sample['label'])
                self.DNNSample = numpy.zeros((self.maxChildCount ** (self.maxDepth - 1),self.maxWordCount))
                self.ConvIndex2Vector(sample['features'], 0, 0, self.maxChildCount ** (self.maxDepth - 1))
                self.DNNSamples.append(self.DNNSample)

        numpy.vstack(self.DNNSamples)
        self.labels = numpy.array(self.labelsList)

        pprint.pprint(self.DNNSamples)