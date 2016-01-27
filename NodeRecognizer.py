from pprint import pprint
import bs4.element
from bs4 import BeautifulSoup

global featureList
featureList = []


class NodeRecognizer:
    #unuse
    def pprint(self, object):
        content = object.toString().split('\n')
        for line in content:
            print(" " * self.level *4 + line)


    def GetNodeType(self, soupNode):
        if (type(soupNode) ==  bs4.element.Tag):
            return 'Tag'
        return None

    def GetNodeContent(self, soupNode):
        if (type(soupNode) ==  bs4.element.NavigableString):
            return soupNode.string
        if (type(soupNode) ==  bs4.element.Tag):
            return soupNode.text

    def GetNodeClass(self, soupNode):
        if (type(soupNode) ==  bs4.element.Tag and 'class' in soupNode.attrs):
            return soupNode['class']
        return []

    def __init__ (self, soupNode, level):
        self.result = None
        self.level = level
        features = {
            'name': soupNode.name,
            'class': self.GetNodeClass(soupNode),
            'childCount': 0,
            'content': self.GetNodeContent(soupNode),
            'type': self.GetNodeType(soupNode),
        }
        self.DNNFeature = []

        if (features['type'] ==  'Tag'):
            self.DNNFeature.append([])
            self.DNNFeature[0].append(features['name'])
            for className in features['class']:
                self.DNNFeature[0].append(className)

            #recursive call
            for child in soupNode.children:
                features['childCount'] += 1
                childRecog = NodeRecognizer(child, self.level + 1)
                if (len(childRecog.DNNFeature) != 0):
                    self.DNNFeature.append(childRecog.DNNFeature)

            
        #tree condition classifier
        if (features['type'] == 'Tag'):
            if (features['childCount'] == 0):
                self.result = features['name']
            if ('item' in features['class']):
                self.result = 'item'
            # data output
            sample = {
                'features': self.DNNFeature
            }
            if self.result == 'item':
                sample['label'] = 1
            else:
                sample['label'] = 0
            featureList.append(sample)

