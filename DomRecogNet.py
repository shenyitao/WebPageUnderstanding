import timeit

import theano.tensor as T
import numpy, theano
import numpy.random
import Layer
import SimpleCrawler
import sys
import os

from ConvLayerForTree import ConvLayerForTree

def BuildModel():
    pass


def DataShuffle(data, randomArray):
    shuffledData = numpy.empty(data.shape)
    count = 0
    for index in randomArray:
        shuffledData[index] = data[count]
        count += 1
    return shuffledData

def DataSplit(dataSet):
    label, features = dataSet[0]
    shape = features.shape
    sampleCount = label.shape[0]
    randomArray = numpy.random.permutation(sampleCount)
    shuffledLabel = DataShuffle(label, randomArray)
    shuffledFeatures = DataShuffle(features, randomArray)


    test_set_x, test_set_y = shared_dataset(dataSet[1][1], dataSet[1][0])
    valid_set_x, valid_set_y = shared_dataset(shuffledFeatures, shuffledLabel)
    train_set_x, train_set_y = shared_dataset(shuffledFeatures, shuffledLabel)

    datasets = [
        (train_set_x, train_set_y),
        (valid_set_x,valid_set_y),
        (test_set_x,test_set_y)
    ]

    return datasets

def shared_dataset(data_x, data_y, borrow=True):
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def TrainModel(learning_rate=0.1,
               n_epochs=20000,
                nkerns=[20, 50],
               batch_size=1):

    ###############
    # TRAIN MODEL #
    ###############

    rng = numpy.random.RandomState(23455)
    dataSet = SimpleCrawler.GetTrainData()

    datasets = DataSplit(dataSet)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches = n_train_batches // batch_size
    n_valid_batches = n_valid_batches // batch_size
    n_test_batches = n_test_batches // batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    # hyper-parameters
    featureNum = 2

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor3('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    inputFeatureCount = 20
    layer0_input =  T.reshape(x,(625, inputFeatureCount))#(node_c * maxChilds_c, feature_c)

    rng = numpy.random.RandomState(213)
    layer0 = ConvLayerForTree(rng = rng,
                                input = layer0_input,
                                input_shape = (625, inputFeatureCount),
                                maxChilds = 5,
                                featureNum = featureNum)

    layer1 = ConvLayerForTree(rng = rng,
                                input = layer0.output,
                                input_shape = (125, featureNum),
                                maxChilds = 5,
                                featureNum = featureNum)


    layer2 = ConvLayerForTree(rng = rng,
                                input = layer1.output,
                                input_shape = (25, featureNum),
                                maxChilds = 5,
                                featureNum = featureNum)


    layer3 = ConvLayerForTree(rng = rng,
                                input = layer2.output,
                                input_shape = (5, featureNum),
                                maxChilds = 5,
                                featureNum = featureNum)


    layer4_input = T.reshape(layer3.output,(featureNum,))

    # construct a fully-connected sigmoidal layer
    layer4 = Layer.HiddenLayer(
        rng,
        input=layer4_input,
        n_in=featureNum,
        n_out=featureNum,
        activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer5 = Layer.LogisticRegression(input=layer4.output, n_in=featureNum, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer5.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    # early-stopping 参数
    patience = 1000000  # 在前patience个 epoh 中，不考虑early-stopping
    patience_increase = 2  # 如果发现了明显更好的，多等待patience_increase－1倍的patience
    improvement_threshold = 0.995 # 验证误差变好超过这个比例才认为是发现了明显更好的
    validation_frequency = min(n_train_batches, patience / 2) # 每validation_frequency轮进行一次验证
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            if (iter + 1) % validation_frequency == 0:

                # 计算0/1验证误差
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %% cost %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.,cost_ij * 100))

                # 如果发现了更好的，
                if this_validation_loss < best_validation_loss:

                    # 如果发现了明显更好的，多等待若干倍patience
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # 保存最好结果
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # 计算测试误差
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                      
if __name__ == '__main__':
    TrainModel()