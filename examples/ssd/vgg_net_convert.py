import caffe

net = caffe.Net('/host-ssd1TB/ssd_vgg/VGG_FACE_16_layers_deploy.prototxt', '/host-ssd1TB/ssd_vgg/VGG_FACE_16_layers.caffemodel', caffe.TEST)

print len(net._layer_names)
print len(net.layers) 
print len(net.params)

params = ['fc6', 'fc7', 'fc8']

for key,val in net.params.iteritems():
	print key

fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
	print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

net_full_conv = caffe.Net('/host-ssd1TB/ssd_vgg/VGG_FACE_16_layers_conv_deploy.prototxt', '/host-ssd1TB/ssd_vgg/VGG_FACE_16_layers.caffemodel', caffe.TEST)

params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']

conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

print conv_params['fc6-conv'][0].shape
print conv_params['fc7-conv'][0].shape

# reshape to fully convolutional layers subsampled to 1024x512x3x3 filter
conv_params['fc6-conv'][0][...] = fc_params['fc6'][0].reshape(4096, 512, 7, 7)[::4, :, ::3, ::3]
conv_params['fc6-conv'][1][...] = fc_params['fc6'][1][::4]

conv_params['fc7-conv'][0][...] = fc_params['fc7'][0].reshape(4096, 4096, 1, 1)[::4, ::4, :, :]
conv_params['fc7-conv'][1][...] = fc_params['fc7'][1][::4]

#for pr, pr_conv in zip(params, params_full_conv):
#    conv_params[pr_conv][0][...] = fc_params[pr][0].reshape(4096, 512, 7, 7)[::4, :, ::3, ::3] #.flat[1::3]  # flat unrolls the arrays
#    conv_params[pr_conv][1][...] = fc_params[pr][1][::4]

print conv_params['fc6-conv'][0].shape
print conv_params['fc6-conv'][0].flatten().shape
print fc_params['fc6'][0].flatten().shape
 
net_full_conv.save('/host-ssd1TB/ssd_vgg/VGG_FACE_16_layers_full_conv.caffemodel')
