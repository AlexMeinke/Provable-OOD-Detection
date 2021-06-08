import os


project_folder = 'YOUR_PROJECT_FOLDER' # this is set in code so that you can make a case distinction if you run the same experiments from different servers/clusters


# Configure these paths to point to the correct locations on your server
location_dict = dict([])
location_dict['CIFAR10'] = 'PATH/CIFAR10'
location_dict['CIFAR100'] = 'PATH/CIFAR100'
location_dict['SVHN'] = 'PATH/SVHN'
location_dict['LSUN'] = 'PATH/LSUN'
location_dict['Flowers'] = 'PATH/flowers/'
location_dict['FGVC'] = 'PATH/FGVC/fgvc-aircraft-2013b/'
location_dict['Cars'] = 'PATH/stanford_cars/'
location_dict['TinyOpen'] = 'PATH/openimages/'
location_dict['TinyImages'] = 'PATH/80M-tiny-images.bin'
location_dict['ImageNet'] = 'PATH/ImageNet2012/'
