# -*- coding: utf-8 -*-

"""
Created on 2016/1/18

@author: shenyitao

ConvLayerForTree
"""
import theano.tensor as T
import numpy, theano
import numpy.random

def relu(x):
    return T.maximum(0,x)
    
    
class ConvLayerForTree (object):
    def __init__ (self, rng, input, input_shape, maxChilds, featureNum=2, activation=relu):

        (node_c, feature_c) = input_shape
        n_out = node_c // maxChilds
        self.input = T.reshape(input,(n_out, maxChilds * feature_c))

        node_n_in = maxChilds * feature_c
        node_n_out = featureNum
        W_value = numpy.asarray(
            rng.uniform(
                low = -1/maxChilds,
                high = 1/maxChilds,
                size = (node_n_in, node_n_out)
                ),
            dtype = theano.config.floatX
            )

        self.W = theano.shared(W_value,name = 'W') #(filters, feature, Fh, Fw)

        b_value = numpy.zeros(
            (n_out, featureNum),
            dtype = theano.config.floatX)
        self.b = theano.shared(b_value, name = 'b')

        vector, updates = theano.scan(fn=lambda childs: T.dot(childs, self.W),
                                  outputs_info=None,
                                  sequences=self.input)

        # store parameters of this layer
        self.params = [self.W, self.b]

        self.activation = activation(vector + self.b)
        self.output = T.reshape(self.activation,(n_out, featureNum))

