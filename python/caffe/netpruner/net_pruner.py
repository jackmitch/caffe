from __future__ import print_function

import math
import os
import shutil
import stat
import subprocess
import sys
import numpy
import scipy
import matplotlib.pyplot as plt
import argparse
import subprocess

#sys.path.insert(0, 'C:\\\\Users\\JLeigh\\MyProjects\\OMGLife\\visp\\caffe\\build\\x64\\Release\\pycaffe')
import caffe
from caffe.proto import caffe_pb2

args = None

def PlotWeightDistribution(name, params):
    print('Display weight distribution for %s' % name)
    weights = params[0]
    y = numpy.reshape(weights.data, (weights.count))
    plt.clf()
    plt.hist(y, bins=100)
    plt.show()
    return


def ClusterLayer(name, params):
    print('clustering layer %s' % name)
    if len(params) != 2:
        return # only handle weights and biases for now

    for blob in params:
        # calculate the distance between all filters
        data = numpy.reshape(blob.data, (blob.shape[0], blob.count / blob.shape[0]))
        dists = numpy.zeros((blob.shape[0], blob.shape[0]))

        for j in xrange(0, blob.shape[0]-1):
            for i in xrange(j+1, blob.shape[0]):
                dists[j][i] = scipy.spatial.distance.sqeuclidean(data[j], data[i])
                dists[i][j] = dists[j][i]
                if dists[j][i] < 0.05:
                    print('%d -> %d = %f' % (j,i,dists[j][i]))

        y = numpy.ndarray.flatten(dists[numpy.triu_indices(dists.shape[0], 1)])
        x = numpy.linspace(0, y.shape[0]-1, y.shape[0])
        plt.scatter(x, y)
        plt.show()
    

def TestAccuracy(solver):
    # calculate the accuracy on the validation set
    accuracy = 0
    loss = 0
    batch_size = solver.test_nets[0].blobs['data'].num
    test_iters = int(len(Xt) / batch_size)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
        loss += solver.test_nets[0].blobs['loss'].data
    accuracy /= test_iters
    loss /= test_iters

    print("Test Accuracy: {:.3f}".format(accuracy))
    return loss, accurancy


def FineTune(solver, tag):
    # fine tune the net
    blobs = ('loss', 'acc')
    loss, acc = ({np.zeros(niter)}
                 for _ in blobs)

    delta = 100
    prev_test_accuracy = 0

    while abs(delta) > 0.005:
        s.step(1)  # run a single SGD step in Caffe

        loss[it], acc[it] = (s.net.blobs[b].data.copy() for b in blobs)

        if it % solver.display == 0 or it + 1 == niter:
            loss_disp = 'loss=%.3f, acc=%2d%%' % (loss[it], np.round(100*acc[it]))
            print('%3d) %s' % (it, loss_disp))
              
        if it % solver.test_interval == 0 or it + 1 == niter:
            test_loss, test_accuracy = TestAccuracy(solver)
            delta = test_accuracy - prev_test_accuracy
            prev_test_accuracy = test_accuracy

    # Save the learned weights from both nets.
    savename = os.path.join(solver.snapshot_prefix, 'weights_%s.caffemodel'%(tag,))
    solver.net.save(savename)
    proto_savename = os.path.join(solver.snapshot_prefix, 'train_val_%s.prototxt'%(tag,))
    sovler.net.save_prototxt(proto_savename)

    return test_loss, test_accuracy


def FineTuneUsingCmdLine(solver, tag):
    # fine tune using the command line
    weights_filename = os.path.join(solver.snapshot_prefix, 'weights_%s.caffemodel'%(tag,))
    solver.net.save(weights_filename)
    cmd = './build/caffe train -solver %s -weights %s -gpu all' % (args.solver_file, weights_filename)
    output = subprocess.check_output(cmd, shell=True)
    proto_savename = os.path.join(solver.snapshot_prefix, 'train_val_%s.prototxt'%(tag,))
    sovler.net.save_prototxt(proto_savename)

    # copy the new weights back into the net
    solver.net.copy_from(weights_filename)

    loss, acc = TestAccuracy(solver)
    return loss, acc

def FindMinChannelWeights(layer):
    min = 9e9
    bc = -1
    
    b = layer.blobs[0]
    bSq = numpy.multiply(b.data, b.data)
    for c in xrange(0, b.data.shape[0]):
        sumSq = numpy.sum(bSq[c,:])
        if sumSq < min:
            min = sumSq
            bc = c

    return min, bc


def TrimBlob(blob, plane, axis):
    new_data = numpy.delete(blob.data, plane, axis)
    new_shape = ()
    for i in xrange(0, len(blob.shape)):
        val = blob.shape[i]
        if i == axis:
            val -= 1
        new_shape = new_shape + (val,)
    blob.reshape(*new_shape)
    numpy.copyto(blob.data, new_data)


def RemoveFilterFromLayer(net, layername, channel):

    # remove the channel from the layer and then shrink all following kernels accordingly
    num_filters = net.params[layername][0].shape[0]
    for params in net.params[layername]:
        TrimBlob(params, channel, 0)

    si = list(net._layer_names).index(layername)
    net.layers[si].alter_params(net, net.params[layername], num_filters=num_filters-1)

    # remove the elements from the kernels in the following conv/InnerProduct layers
    for i in xrange(si+1, len(net._layer_names)):
        if net.layers[i].type == 'Convolution':
            params = net.layers[i].blobs[0]

            # check if the input blob does not match the params blob, if it doesn't
            # then this conv layer was affected by the removal of the channel and we 
            # need to clip it's params
            for b0 in net._bottom_ids(i):
                if net._blobs[b0].shape[1] != params.shape[1]:
                    #print('trimming layer %s'%net._layer_names[i])
                    TrimBlob(params, channel, 1)
                    net.layers[i].alter_params(net, net.layers[i].blobs, num_channels=num_filters-1)
                    break

        elif net.layers[i].type == 'InnerProduct':
            # check is connected directly to conv layer
            conv_link = False
            exit = False
            targets = list(net._bottom_ids(i))
            # cycle back through layers to find if this layer is connected to another InnerProduct
            for j in xrange(i-1, 0, -1):
                for b0 in net._top_ids(j):
                    if b0 in targets:
                        # this layer feeds into top
                        if net.layers[j].type == 'InnerProduct':
                            # Inner product is fed by another inner product so don't need to update params
                            exit = True
                            break
                        if net.layers[j].type == 'Convolution':
                           # if innerproduct is fed by the conv layer we just changed then we need to modify the innerproduct params
                           conv_link = (j == si)
                           exit = True
                           break
                    targets.append(b0)

                    if exit:
                        break
                if exit:
                    break
            
            if conv_link:
                b = net.layers[i].blobs[0]
                count = 1
                input_shape = None
                input_blob_ids = net._bottom_ids(i)
                if len(input_blob_ids) != 1:
                    raise ValueError('Cannot resize InnerProduct if there are more than one input blob')
                input_blob = net._blobs[input_blob_ids[0]]
                if input_blob != None:
                    input_shape = input_blob.shape
                else:
                    raise ValueError('Could not find input blob for fully connected layer')
                csize = input_shape[2] * input_shape[3]
                new_data = numpy.delete(b.data, range(csize * channel, csize * (channel+1)), 1)
                new_shape = (b.shape[0], b.shape[1] - csize)
                b.reshape(*new_shape)
                numpy.copyto(b.data, new_data)
                net.layers[i].alter_params(net, net.layers[i].blobs, num_channels=num_filters-1)
        else:
            net.layers[i].alter_params(net, net.layers[i].blobs, num_channels=num_filters-1)

    # reshape to make sure everything agrees    
    net.reshape()


def CountNumParams(net):
    count = 0
    for (name, params) in net.params.iteritems():
        for blob in params:
            count += blob.count
    return count


def PruneChannels(solver, weights):

    solver.net.copy_from(weights)

    start_size = CountNumParams(solver.net)
    comp_ratio = 1.0

    f = open('channel_pruning.txt', 'w')
    f.write('compression_ratio,num_params,pre-fine-tune-accuracy,post-fine-tune-accuracy\n')
    f.write('0,%.0f,0,0\n'%start_size)
    print('Starting param size %.0f\n'%start_size)

    # for all convolutional layers look for the channels that are closest
    more = True
    while more:

        layer_to_prune = []
        chan_to_prune = []
        start_comp_ratio = comp_ratio

        # reduce the net by 5% 
        while start_comp_ratio - comp_ratio < 0.05:
            # find the best channel to prune, for now the smallest abs weights
            min_sum = 9.9e9
            bi = -1
            bc = 0

            for i in xrange(0, len(solver.net.layers)):
                if solver.net.layers[i].type == 'Convolution':
                    #PlotWeightDistribution(name, solver.net.params[name])
                    sum, c = FindMinChannelWeights(solver.net.layers[i])
                    if sum < min_sum:
                        min_sum = sum
                        bi = i
                        bc = c
        
            print('Removing channel %d from layer %s with score %f' % (bc, solver.net._layer_names[bi], min_sum))

            RemoveFilterFromLayer(solver.net, solver.net._layer_names[bi], bc)

            new_size = CountNumParams(solver.net)
            comp_ratio = float(new_size) / start_size
    
        # fine tune the net
        pre_acc = TestAccuracy()
        loss, post_acc = FineTuneUsingCmdLine # FineTune(solver, str(comp_ratio))

        f.write('%.3f,%.0f,%.3f,%.3f\n'%(comp_ratio, new_size, pre_acc, post_acc)) 
        f.flush()
        os.fsync()
        print('compression ratio: %.3f (new size: %.0f) pre-accuracy: %.3f post-accuracy: %.3f\n'%(comp_ratio, new_size, pre_acc, post_acc))

    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--modify-type',
        dest="modify_type",
        help="Type of net modification to be performed",
        default="prune-channels")

    parser.add_argument(
        '--solver-file',
        dest="solver_file",
        help="solver file to use for finetuning")

    parser.add_argument(
        '--weights-file',
        dest="weights_file",
        help="weights file for the net used in hte solver")

    parser.add_argument(
        '--cpu',
        dest="cpu",
        help="run on cpu or gpu",
        default = True)

    args = parser.parse_args()

    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()

    solver = None
    if len(args.solver_file) > 0:
        solver = caffe.get_solver(args.solver_file)

    if args.modify_type == 'prune-channels':
        #try:
            PruneChannels(solver, args.weights_file)
        #except Exception as e:
        #    print(e)
    else:
        print("Error: Unknown modification type");
