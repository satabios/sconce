===========================
Pipeline-Tutorial
===========================

.. code:: ipython3

    import timm
    from torchsummary import summary
    import numpy as np
    from datetime import datetime 
    
    import torch
    import torch.nn as nn 
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import collections
    from collections import defaultdict
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import ipdb
    import timm
    from torchvision.datasets import CIFAR10
    import torch_pruning as tp
    import torchvision.models as models
    import time
    
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    
    import brevitas.nn as qnn
    
    
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
    )
    warnings.filterwarnings(
        action='default',
        module=r'torch.ao.quantization'
    )
    
    # Specify random seed for repeatable results
    torch.manual_seed(191009)
    
    
    
    
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class config:
        lr = 1e-4
        n_classes = 10
        epochs = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = 64

.. code:: ipython3

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    train_dataset = CIFAR10(root='data/', download=True, transform=transform_train)
    valid_dataset = CIFAR10(root='data/',  download=True,train=False, transform=transform_test)
    
    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)
    
    



.. parsed-literal::

    Files already downloaded and verified
    Files already downloaded and verified


.. code:: ipython3

    
    
    def training(model, train_loader, valid_loader):
        
        optim = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()
        model = model.to(config.device)
        
        model.train()
        validation_acc = 0
        ########### Train  ###############
        for ep in range(config.epochs):
            running_loss = 0
            running_acc = 0
            for batch_idx, data in enumerate(train_loader):
                optim.zero_grad()
                image, label = data
                image, label = image.to(config.device), label.to(config.device)
                out = model(image)
                loss = criterion(out, label)
                acc = torch.argmax(out, 1) - label
                running_acc+= len(acc[acc==0])
                running_loss+= loss.item() * label.size(0)
                loss.backward()
                optim.step()
            epoch_loss = running_loss/ len(train_loader.dataset)
            train_epoch_acc = running_acc/ len(train_loader.dataset)
            
            
            ########### Validate #############
    
    
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                      f'Epoch: {ep+1}\t'
                      f'Train loss: {epoch_loss:.4f}\t'
                      f'Train accuracy: {100 * train_epoch_acc:.2f}\t')
            val_acc = validate(model, valid_loader)
            if(validation_acc< val_acc):
                validation_acc = val_acc
                torch.save(model.state_dict(), './weights/dl-x25'+str(validation_acc)+'.pth')
                
        print("Final Validation Accuracy:", validation_acc, "\n \n")
        return model
    
    def validate(model, valid_loader):
        val_running_acc = 0
        model.eval()
        for batch_idx, data in enumerate(valid_loader):
    
            image, label = data
            image, label = image.to(config.device), label.to(config.device)
            out = model(image)
            acc = torch.argmax(out, 1) - label
            val_running_acc+= len(acc[acc==0])
        val_epoch_acc = val_running_acc/ len(valid_loader.dataset)
    
    
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Valid accuracy: {100 * val_epoch_acc:.2f}')
        return val_epoch_acc
            
                


.. code:: ipython3

    number_conv_layers = 0
    
    
    def universal_layer_identifier(identification, module, module_name):
        if(identification == None):
            if isinstance(module, nn.Conv2d):
                return True
        else:
            if (identification in module_name):
                return True
    def universal_get_layer_id_pruning(model, pruning_type, pruning_percent, identification=None): #identification = none
        if(pruning_type == 'L1'):
            strategy  = tp.strategy.L1Strategy()
        elif(pruning_type == 'L2'):
            strategy  = tp.strategy.L2Strategy()
        channels_pruned = []
        
        def find_instance(obj):
            if isinstance(obj, nn.Conv2d):
                pruning_idx = strategy(obj.weight, amount = pruning_percent)
                channels_pruned.append(pruning_idx)
                return None
            elif isinstance(obj, list):
                for internal_obj in obj:
                    find_instance(internal_obj)
            elif (hasattr(obj, '__class__')):
                for internal_obj in obj.children():
                    find_instance(internal_obj)
            elif isinstance(obj, OrderedDict):
                for key, value in obj.items():
                    find_instance(value)
    
        find_instance(model)
    
        channels_pruned = np.asarray(channels_pruned, dtype=object)
        return channels_pruned
    
    
    
    def universal_filter_pruning(model, input_shape, channels_pruned, identification=None):
        DG = tp.DependencyGraph()
        DG.build_dependency(model, example_inputs= torch.randn(input_shape).to(config.device))
    
        layer_id = 0
    
        def find_instance(obj):
            if isinstance(obj, nn.Conv2d):
                global number_conv_layers
                number_conv_layers+=obj.out_channels
                pruning_plan = DG.get_pruning_plan(obj, tp.prune_conv_out_channel, idxs=channels_pruned[layer_id])
                pruning_plan.exec()
                return None
            elif isinstance(obj, list):
                for internal_obj in obj:
                    find_instance(internal_obj)
            elif (hasattr(obj, '__class__')):
                for internal_obj in obj.children():
                    find_instance(internal_obj)
            elif isinstance(obj, OrderedDict):
                for key, value in obj.items():
                    find_instance(value)
    
        find_instance(model)
        return model
    
    
    
    
    
    


.. code:: ipython3

    from thop import profile, clever_format
    
    
    def pruner(model,experiment_name, config, input_dims, pruning_stratergy, pruning_percent,  train_loader, valid_loader):
        original_model = model
        input = torch.randn((config.batch_size, )+ input_dims).to(config.device)
        
        macs_original, params_original = profile(original_model, inputs=(input, ))
    #     macs_original, params_original = clever_format([macs_original, params_original], "%.3f")
        
        torch.save(original_model.state_dict(), './weights/'+experiment_name+'.pth')
        print("\n \n Original Validation Accuracy: \n \n")
        validate(model, valid_loader)
        
        channels_pruned = universal_get_layer_id_pruning(model, pruning_stratergy, pruning_percent)
        print("\n \n ################################# Post Purning ################################# \n \n ")
        print("Original Conv Layers in the Model:", number_conv_layers ,"\n Number of Layers Selected:", len(channels_pruned), "\n Number of Filters Pruned:",sum([len(x) for x in channels_pruned]))
        pruned_model = universal_filter_pruning(model, (config.batch_size,)+input_dims, channels_pruned).to(config.device)
        pruned_model = training(pruned_model, train_loader, valid_loader)
        torch.save(pruned_model, './weights/'+experiment_name+'_pruned_model.pth')
        torch.save(pruned_model.state_dict(), './weights/'+experiment_name+'_pruned.pth')
        print("\n \n Pruned Validation Accuracy: \n \n")
        validate(pruned_model, valid_loader)
        
        print("\n \n ################################# MAC's and Parameters Comparison #################################")
    
        
        macs_pruned, params_pruned = profile(pruned_model, inputs=(input, ))
    
        print("\n \n Original Model MAC's and Params:",macs_original, params_original)
        print("Pruned Model MAC's and Params:",macs_pruned, params_pruned )
    
    


.. code:: ipython3

    
    resnet = timm.create_model('resnet18', num_classes=10).to("cuda")
      # Pretrained Model
    resnet.load_state_dict(torch.load('/home/beast/Downloads/resnet18.pth'))
    
    
    pruner(resnet,"resnet18", config,(3,32,32), "L2", 0.06,  train_loader, valid_loader)


.. parsed-literal::

    [INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
    [INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
    [INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
    [INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool2d'>.
    [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
    [INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>.
    [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
    
     
     Original Validation Accuracy: 
     
    
    20:32:28 --- Valid accuracy: 72.75
    
     
     ################################# Post Purning ################################# 
     
     
    Original Conv Layers in the Model: 4764 
     Number of Layers Selected: 20 
     Number of Filters Pruned: 275
    20:32:48 --- Epoch: 1	Train loss: 0.9083	Train accuracy: 68.01	
    20:32:50 --- Valid accuracy: 72.65
    20:33:11 --- Epoch: 2	Train loss: 0.7704	Train accuracy: 72.79	
    20:33:13 --- Valid accuracy: 73.75
    Final Validation Accuracy: 0.7375 
     
    
    
     
     Pruned Validation Accuracy: 
     
    
    20:33:15 --- Valid accuracy: 73.75
    
     
     ################################# MAC's and Parameters Comparison #################################
    [INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
    [INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
    [INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
    [INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool2d'>.
    [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
    [INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>.
    [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
    
     
     Original Model MAC's and Params: 2382102528.0 11181642.0
    Pruned Model MAC's and Params: 2152319040.0 10769376.0


.. code:: ipython3

                                  ######### Original Model  #########
    # original_model = timm.create_model('resnet18', num_classes=10).to("cuda")
    # original_model.load_state_dict(torch.load('./weights/resnet18.pth'), strict=False)
    # validate(original_model, valid_loader)
    # summary(original_model,(3,32,32))
    


.. code:: ipython3

                                    ######### Pruned Model  #########
    # pruned_model = torch.load("./weights/resnet18_pruned_model.pth")
    # pruned_model.load_state_dict(torch.load('./weights/resnet18_pruned.pth'), strict=False)
    # validate(pruned_model, valid_loader)
    # summary(pruned_model,(3,32,32))
        

.. code:: ipython3

    pruned_model = torch.load("./weights/resnet18_pruned_model.pth").to("cuda")
    pruned_model.eval()
    





.. parsed-literal::

    ResNet(
      (conv1): Conv2d(3, 55, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(55, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(55, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(61, 55, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(55, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(55, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(61, 55, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(55, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(55, 125, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(125, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(125, 119, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(55, 119, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(119, 125, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(125, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(125, 119, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(119, 253, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(253, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(253, 247, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(119, 247, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(247, 253, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(253, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(253, 247, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(247, 509, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(509, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(509, 503, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(503, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(247, 503, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(503, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(503, 509, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(509, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(509, 503, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(503, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
      (fc): Linear(in_features=503, out_features=10, bias=True)
    )



.. code:: ipython3

                                  ######## Pick the layers to fuse ########
    layer_list = []
    layer_fuse_list= []
    flag =-1
    
    for name, layer in pruned_model.named_modules():
        
        if(isinstance(layer, nn.ReLU)):
                
            layer_fuse_list.append(name)
            layer_list.append(layer_fuse_list)
            layer_fuse_list = []
    
        if ((len(layer_fuse_list)<2) and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d))):
            
            layer_fuse_list.append(name)
            
            
    
    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    pruned_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    pruned_model_prepared = torch.quantization.prepare_qat(pruned_model.train())

.. code:: ipython3

    pruned_model_prepared = training(pruned_model_prepared, train_loader, valid_loader)


.. parsed-literal::

    14:26:37 --- Epoch: 1	Train loss: 0.7816	Train accuracy: 72.40	
    14:26:40 --- Valid accuracy: 74.32
    14:27:03 --- Epoch: 2	Train loss: 0.7209	Train accuracy: 74.48	
    14:27:06 --- Valid accuracy: 75.23
    Final Validation Accuracy: 0.7523 
     
    


.. code:: ipython3

    
    
    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, fuses modules where appropriate,
    # and replaces key operators with quantized implementations.
    pruned_model_prepared = pruned_model_prepared.to("cpu")
    pruned_model_prepared.eval()
    





.. parsed-literal::

    ResNet(
      (conv1): Conv2d(
        3, 55, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
          fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0019, 0.0009, 0.0012, 0.0014, 0.0014, 0.0015, 0.0015, 0.0017, 0.0011,
                  0.0014, 0.0012, 0.0011, 0.0014, 0.0008, 0.0017, 0.0014, 0.0012, 0.0011,
                  0.0016, 0.0015, 0.0012, 0.0017, 0.0014, 0.0008, 0.0011, 0.0014, 0.0016,
                  0.0011, 0.0016, 0.0017, 0.0013, 0.0015, 0.0018, 0.0017, 0.0013, 0.0010,
                  0.0014, 0.0017, 0.0015, 0.0011, 0.0015, 0.0016, 0.0015, 0.0018, 0.0017,
                  0.0015, 0.0013, 0.0020, 0.0011, 0.0014, 0.0015, 0.0013, 0.0012, 0.0011,
                  0.0014]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
          (activation_post_process): MovingAveragePerChannelMinMaxObserver(
            min_val=tensor([-0.2485, -0.0729, -0.1495, -0.1853, -0.1816, -0.1421, -0.1910, -0.1804,
                    -0.1371, -0.1842, -0.1547, -0.1389, -0.1847, -0.1082, -0.2192, -0.1436,
                    -0.1081, -0.1412, -0.1731, -0.1427, -0.1596, -0.1550, -0.1654, -0.1027,
                    -0.1252, -0.1434, -0.2046, -0.1335, -0.1905, -0.2197, -0.1633, -0.1701,
                    -0.1528, -0.2232, -0.1637, -0.1294, -0.1813, -0.1731, -0.1557, -0.1426,
                    -0.1538, -0.1243, -0.1726, -0.1406, -0.1134, -0.1171, -0.1715, -0.2182,
                    -0.1398, -0.1437, -0.1303, -0.1693, -0.1484, -0.1105, -0.1595]), max_val=tensor([0.1818, 0.1167, 0.1056, 0.1589, 0.1451, 0.1926, 0.1825, 0.2153, 0.1303,
                    0.1551, 0.1161, 0.1417, 0.1728, 0.1036, 0.1938, 0.1807, 0.1471, 0.1331,
                    0.1996, 0.1884, 0.1389, 0.2136, 0.1739, 0.0934, 0.1359, 0.1833, 0.1396,
                    0.1406, 0.1983, 0.1344, 0.1410, 0.1887, 0.2321, 0.1728, 0.1406, 0.1319,
                    0.1735, 0.2127, 0.1954, 0.1131, 0.1872, 0.2043, 0.1875, 0.2251, 0.2199,
                    0.1868, 0.1548, 0.2572, 0.1241, 0.1792, 0.1864, 0.1709, 0.1307, 0.1378,
                    0.1740])
          )
        )
        (activation_post_process): FusedMovingAvgObsFakeQuantize(
          fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.2051]), zero_point=tensor([63], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-12.982280731201172, max_val=13.066455841064453)
        )
      )
      (bn1): BatchNorm2d(
        55, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        (activation_post_process): FusedMovingAvgObsFakeQuantize(
          fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1542]), zero_point=tensor([61], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-9.443758964538574, max_val=10.135025024414062)
        )
      )
      (act1): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(
            55, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0020, 0.0026, 0.0025, 0.0030, 0.0027, 0.0036, 0.0024, 0.0028, 0.0023,
                      0.0022, 0.0026, 0.0027, 0.0023, 0.0043, 0.0029, 0.0024, 0.0028, 0.0042,
                      0.0023, 0.0024, 0.0031, 0.0029, 0.0028, 0.0032, 0.0026, 0.0021, 0.0027,
                      0.0025, 0.0022, 0.0024, 0.0024, 0.0029, 0.0028, 0.0028, 0.0025, 0.0029,
                      0.0029, 0.0030, 0.0032, 0.0022, 0.0022, 0.0028, 0.0032, 0.0028, 0.0031,
                      0.0030, 0.0033, 0.0025, 0.0024, 0.0027, 0.0032, 0.0040, 0.0030, 0.0023,
                      0.0029, 0.0026, 0.0033, 0.0029, 0.0023, 0.0028, 0.0026]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2606, -0.2901, -0.2949, -0.3866, -0.3512, -0.4570, -0.3134, -0.3039,
                        -0.2970, -0.2757, -0.3338, -0.3425, -0.2585, -0.5459, -0.3518, -0.3125,
                        -0.3642, -0.5385, -0.2938, -0.3060, -0.3947, -0.3715, -0.3622, -0.4089,
                        -0.3294, -0.2693, -0.2845, -0.3094, -0.2877, -0.3086, -0.3066, -0.3680,
                        -0.3576, -0.3520, -0.3227, -0.3765, -0.3140, -0.3817, -0.3428, -0.2792,
                        -0.2694, -0.3594, -0.4114, -0.3453, -0.3946, -0.3793, -0.4283, -0.2960,
                        -0.3131, -0.3491, -0.3678, -0.5103, -0.3799, -0.2933, -0.3683, -0.3302,
                        -0.4252, -0.3474, -0.2933, -0.3196, -0.3332]), max_val=tensor([0.1649, 0.3242, 0.3199, 0.2692, 0.3199, 0.2600, 0.2928, 0.3562, 0.2540,
                        0.2835, 0.2495, 0.3167, 0.2951, 0.2782, 0.3637, 0.2951, 0.2474, 0.2982,
                        0.2871, 0.2621, 0.3453, 0.2335, 0.2366, 0.3018, 0.3112, 0.2671, 0.3391,
                        0.3159, 0.2499, 0.2373, 0.2771, 0.3093, 0.2772, 0.2522, 0.2724, 0.2499,
                        0.3636, 0.3178, 0.4124, 0.2464, 0.2768, 0.2581, 0.2613, 0.3543, 0.2287,
                        0.3329, 0.2426, 0.3119, 0.2802, 0.2640, 0.4079, 0.2569, 0.2512, 0.2414,
                        0.2213, 0.3118, 0.2687, 0.3700, 0.2940, 0.3558, 0.2605])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.5544]), zero_point=tensor([79], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-44.02864074707031, max_val=26.383365631103516)
            )
          )
          (bn1): BatchNorm2d(
            61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1017]), zero_point=tensor([69], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-6.967789173126221, max_val=5.947970390319824)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            61, 55, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0025, 0.0029, 0.0033, 0.0023, 0.0026, 0.0022, 0.0032, 0.0028, 0.0032,
                      0.0031, 0.0026, 0.0028, 0.0030, 0.0029, 0.0030, 0.0023, 0.0029, 0.0025,
                      0.0021, 0.0025, 0.0023, 0.0027, 0.0022, 0.0031, 0.0031, 0.0024, 0.0025,
                      0.0033, 0.0026, 0.0026, 0.0027, 0.0033, 0.0030, 0.0019, 0.0031, 0.0027,
                      0.0029, 0.0028, 0.0026, 0.0024, 0.0030, 0.0020, 0.0019, 0.0027, 0.0024,
                      0.0033, 0.0028, 0.0033, 0.0023, 0.0028, 0.0029, 0.0024, 0.0018, 0.0029,
                      0.0026]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.3260, -0.3144, -0.4208, -0.2889, -0.3387, -0.2802, -0.4036, -0.3085,
                        -0.4072, -0.3971, -0.3236, -0.3600, -0.3895, -0.3749, -0.3832, -0.2634,
                        -0.3724, -0.3213, -0.2637, -0.2388, -0.1943, -0.2993, -0.2514, -0.3630,
                        -0.3914, -0.2804, -0.3199, -0.4193, -0.3338, -0.2294, -0.3411, -0.2777,
                        -0.3859, -0.2345, -0.3824, -0.3453, -0.2872, -0.3524, -0.2963, -0.3046,
                        -0.3877, -0.2541, -0.2390, -0.2421, -0.3108, -0.4213, -0.3558, -0.3442,
                        -0.2486, -0.3627, -0.3460, -0.3123, -0.1908, -0.3686, -0.3302]), max_val=tensor([0.2191, 0.3693, 0.2660, 0.2592, 0.2806, 0.2856, 0.2925, 0.3563, 0.3608,
                        0.2640, 0.3302, 0.2472, 0.2453, 0.3033, 0.3152, 0.2864, 0.2712, 0.2283,
                        0.2230, 0.3175, 0.2938, 0.3433, 0.2796, 0.3910, 0.2861, 0.3034, 0.2593,
                        0.3280, 0.3193, 0.3258, 0.2621, 0.4180, 0.2724, 0.2448, 0.3955, 0.2618,
                        0.3695, 0.2465, 0.3250, 0.2996, 0.2946, 0.1969, 0.2157, 0.3439, 0.2857,
                        0.3225, 0.3237, 0.4228, 0.2930, 0.2594, 0.3719, 0.2640, 0.2308, 0.3531,
                        0.2566])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.2990]), zero_point=tensor([74], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-22.19329833984375, max_val=15.78257942199707)
            )
          )
          (bn2): BatchNorm2d(
            55, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0165]), zero_point=tensor([67], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.1005405187606812, max_val=0.9941551685333252)
            )
          )
          (act2): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(
            55, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0022, 0.0034, 0.0028, 0.0036, 0.0028, 0.0025, 0.0026, 0.0042, 0.0033,
                      0.0027, 0.0037, 0.0041, 0.0020, 0.0028, 0.0023, 0.0028, 0.0026, 0.0041,
                      0.0028, 0.0033, 0.0036, 0.0024, 0.0038, 0.0026, 0.0023, 0.0033, 0.0030,
                      0.0026, 0.0032, 0.0025, 0.0032, 0.0025, 0.0022, 0.0030, 0.0031, 0.0022,
                      0.0039, 0.0027, 0.0037, 0.0032, 0.0031, 0.0027, 0.0031, 0.0030, 0.0036,
                      0.0034, 0.0029, 0.0028, 0.0031, 0.0036, 0.0041, 0.0029, 0.0039, 0.0038,
                      0.0023, 0.0039, 0.0034, 0.0026, 0.0026, 0.0033, 0.0033]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2782, -0.4401, -0.3558, -0.4559, -0.3585, -0.3230, -0.3296, -0.5398,
                        -0.4200, -0.3508, -0.4751, -0.5216, -0.2595, -0.2581, -0.2909, -0.3573,
                        -0.2990, -0.2631, -0.3298, -0.4281, -0.4639, -0.3017, -0.4868, -0.3010,
                        -0.2946, -0.4210, -0.3893, -0.3372, -0.4137, -0.2908, -0.4150, -0.3112,
                        -0.2866, -0.3848, -0.4030, -0.2835, -0.4989, -0.3518, -0.4776, -0.4080,
                        -0.3990, -0.3436, -0.3037, -0.3782, -0.3562, -0.2983, -0.3763, -0.2642,
                        -0.3574, -0.4643, -0.5221, -0.3083, -0.5044, -0.4835, -0.2988, -0.5022,
                        -0.4400, -0.3381, -0.3187, -0.4178, -0.3774]), max_val=tensor([0.2655, 0.2372, 0.3398, 0.2746, 0.3185, 0.2693, 0.2335, 0.3073, 0.2795,
                        0.1963, 0.3693, 0.2991, 0.2341, 0.3558, 0.2478, 0.2381, 0.3254, 0.5151,
                        0.3525, 0.2419, 0.3070, 0.2823, 0.3285, 0.3261, 0.2864, 0.3021, 0.2463,
                        0.3180, 0.3062, 0.3221, 0.2967, 0.3187, 0.1889, 0.2836, 0.3150, 0.2760,
                        0.2558, 0.2216, 0.2790, 0.2869, 0.2967, 0.3132, 0.3879, 0.2712, 0.4592,
                        0.4323, 0.2847, 0.3563, 0.3940, 0.3776, 0.3981, 0.3662, 0.2540, 0.2504,
                        0.2515, 0.2675, 0.2610, 0.2997, 0.3319, 0.3643, 0.4177])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.5178]), zero_point=tensor([79], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-41.09857177734375, max_val=24.664838790893555)
            )
          )
          (bn1): BatchNorm2d(
            61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0991]), zero_point=tensor([68], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-6.755105018615723, max_val=5.830554962158203)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            61, 55, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0029, 0.0030, 0.0031, 0.0023, 0.0030, 0.0030, 0.0031, 0.0030, 0.0029,
                      0.0028, 0.0033, 0.0027, 0.0017, 0.0033, 0.0034, 0.0031, 0.0020, 0.0024,
                      0.0035, 0.0025, 0.0018, 0.0034, 0.0017, 0.0029, 0.0025, 0.0026, 0.0030,
                      0.0034, 0.0029, 0.0025, 0.0025, 0.0030, 0.0030, 0.0029, 0.0034, 0.0026,
                      0.0029, 0.0023, 0.0028, 0.0029, 0.0026, 0.0031, 0.0026, 0.0028, 0.0024,
                      0.0030, 0.0036, 0.0030, 0.0027, 0.0032, 0.0028, 0.0025, 0.0024, 0.0027,
                      0.0030]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.3660, -0.2944, -0.3997, -0.2919, -0.3777, -0.3839, -0.3277, -0.3253,
                        -0.3496, -0.3582, -0.3223, -0.3433, -0.2189, -0.3854, -0.4367, -0.2646,
                        -0.2518, -0.2559, -0.4450, -0.2525, -0.2274, -0.3011, -0.2202, -0.3747,
                        -0.2933, -0.3131, -0.3783, -0.4232, -0.3150, -0.3010, -0.2985, -0.3358,
                        -0.3799, -0.3661, -0.3483, -0.3288, -0.3742, -0.2892, -0.2287, -0.3764,
                        -0.3346, -0.3967, -0.2934, -0.3605, -0.2908, -0.3011, -0.3375, -0.2846,
                        -0.3396, -0.3190, -0.3580, -0.3151, -0.2990, -0.3500, -0.3825]), max_val=tensor([0.3030, 0.3804, 0.3181, 0.2969, 0.3567, 0.3370, 0.3902, 0.3785, 0.3684,
                        0.2933, 0.4182, 0.2459, 0.1895, 0.4136, 0.3572, 0.3928, 0.2505, 0.2997,
                        0.3182, 0.3236, 0.2146, 0.4264, 0.2129, 0.2529, 0.3235, 0.3259, 0.2987,
                        0.4278, 0.3639, 0.3130, 0.3158, 0.3787, 0.2694, 0.2705, 0.4277, 0.2775,
                        0.2972, 0.2873, 0.3570, 0.2847, 0.3043, 0.2995, 0.3346, 0.2570, 0.3020,
                        0.3800, 0.4522, 0.3805, 0.2803, 0.4103, 0.3417, 0.2436, 0.3067, 0.3393,
                        0.2623])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.3395]), zero_point=tensor([59], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-19.893938064575195, max_val=23.228788375854492)
            )
          )
          (bn2): BatchNorm2d(
            55, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0197]), zero_point=tensor([63], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.2366790771484375, max_val=1.267777681350708)
            )
          )
          (act2): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(
            55, 125, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0031, 0.0028, 0.0029, 0.0035, 0.0026, 0.0035, 0.0039, 0.0021, 0.0022,
                      0.0050, 0.0027, 0.0024, 0.0028, 0.0042, 0.0035, 0.0036, 0.0041, 0.0025,
                      0.0048, 0.0042, 0.0028, 0.0037, 0.0029, 0.0025, 0.0020, 0.0027, 0.0024,
                      0.0041, 0.0034, 0.0026, 0.0035, 0.0024, 0.0019, 0.0026, 0.0040, 0.0045,
                      0.0027, 0.0025, 0.0036, 0.0038, 0.0030, 0.0030, 0.0020, 0.0028, 0.0030,
                      0.0030, 0.0036, 0.0030, 0.0024, 0.0034, 0.0035, 0.0026, 0.0027, 0.0022,
                      0.0030, 0.0031, 0.0030, 0.0036, 0.0048, 0.0025, 0.0024, 0.0031, 0.0030,
                      0.0029, 0.0032, 0.0026, 0.0032, 0.0030, 0.0031, 0.0033, 0.0037, 0.0027,
                      0.0033, 0.0032, 0.0041, 0.0026, 0.0047, 0.0049, 0.0025, 0.0026, 0.0037,
                      0.0025, 0.0030, 0.0024, 0.0025, 0.0028, 0.0026, 0.0035, 0.0034, 0.0026,
                      0.0022, 0.0023, 0.0038, 0.0024, 0.0029, 0.0044, 0.0034, 0.0026, 0.0037,
                      0.0025, 0.0034, 0.0031, 0.0029, 0.0030, 0.0034, 0.0039, 0.0021, 0.0026,
                      0.0031, 0.0027, 0.0034, 0.0034, 0.0049, 0.0023, 0.0029, 0.0042, 0.0028,
                      0.0049, 0.0023, 0.0029, 0.0043, 0.0026, 0.0024, 0.0026, 0.0034]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.3913, -0.2760, -0.3730, -0.4436, -0.3367, -0.4497, -0.4979, -0.2524,
                        -0.2823, -0.6362, -0.3419, -0.3080, -0.3629, -0.5379, -0.4491, -0.4614,
                        -0.5187, -0.3160, -0.6185, -0.4866, -0.3575, -0.3579, -0.3675, -0.3199,
                        -0.2577, -0.3408, -0.3105, -0.5302, -0.3244, -0.2868, -0.3037, -0.2777,
                        -0.2454, -0.3245, -0.5067, -0.5748, -0.3403, -0.3255, -0.4587, -0.4829,
                        -0.3813, -0.3719, -0.2427, -0.3426, -0.2946, -0.3853, -0.3505, -0.3843,
                        -0.3103, -0.4346, -0.4435, -0.3281, -0.3282, -0.2821, -0.3903, -0.3968,
                        -0.3440, -0.4585, -0.6118, -0.3115, -0.3133, -0.3921, -0.3812, -0.3684,
                        -0.4134, -0.3274, -0.4104, -0.3795, -0.3945, -0.4276, -0.4716, -0.3483,
                        -0.4270, -0.4099, -0.5291, -0.3111, -0.5994, -0.6317, -0.3185, -0.3388,
                        -0.3038, -0.3247, -0.3685, -0.2794, -0.3202, -0.3622, -0.3342, -0.4527,
                        -0.4327, -0.3133, -0.2676, -0.2920, -0.3332, -0.3065, -0.3763, -0.5654,
                        -0.4298, -0.2601, -0.4734, -0.3244, -0.4376, -0.3927, -0.3717, -0.2763,
                        -0.4350, -0.5029, -0.2722, -0.3270, -0.3967, -0.3514, -0.4295, -0.4395,
                        -0.6300, -0.2947, -0.3772, -0.5339, -0.3467, -0.3794, -0.2932, -0.3681,
                        -0.5471, -0.3331, -0.3053, -0.3275, -0.4310]), max_val=tensor([0.3415, 0.3507, 0.2752, 0.3718, 0.3005, 0.3323, 0.2539, 0.2647, 0.2842,
                        0.2844, 0.2058, 0.2727, 0.2706, 0.2493, 0.3073, 0.2833, 0.2569, 0.2403,
                        0.3493, 0.5305, 0.2646, 0.4704, 0.3431, 0.2067, 0.2423, 0.2833, 0.2289,
                        0.2406, 0.4304, 0.3349, 0.4488, 0.3046, 0.2228, 0.3330, 0.3993, 0.4105,
                        0.2853, 0.2884, 0.2942, 0.4044, 0.2486, 0.3757, 0.2493, 0.3591, 0.3766,
                        0.2330, 0.4612, 0.2463, 0.1765, 0.2778, 0.2386, 0.2938, 0.3377, 0.2557,
                        0.2292, 0.3295, 0.3846, 0.3339, 0.2326, 0.3202, 0.2564, 0.3631, 0.3097,
                        0.3096, 0.3160, 0.3270, 0.2533, 0.2830, 0.2205, 0.3100, 0.2345, 0.3323,
                        0.2110, 0.2831, 0.2533, 0.3336, 0.1985, 0.2883, 0.2697, 0.2605, 0.4666,
                        0.3058, 0.3822, 0.2994, 0.2351, 0.3028, 0.3210, 0.2688, 0.2102, 0.3260,
                        0.2744, 0.2832, 0.4827, 0.2891, 0.3349, 0.2484, 0.2334, 0.3261, 0.2369,
                        0.2526, 0.2685, 0.2390, 0.3135, 0.3857, 0.3497, 0.2550, 0.2065, 0.2610,
                        0.3528, 0.2283, 0.3746, 0.4023, 0.2913, 0.2373, 0.2790, 0.2725, 0.3517,
                        0.6182, 0.2959, 0.3180, 0.2375, 0.2249, 0.1643, 0.2595, 0.2923])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.6230]), zero_point=tensor([77], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-48.1874885559082, max_val=30.933076858520508)
            )
          )
          (bn1): BatchNorm2d(
            125, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1188]), zero_point=tensor([64], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-7.63934850692749, max_val=7.446564197540283)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            125, 119, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0028, 0.0031, 0.0029, 0.0030, 0.0028, 0.0027, 0.0027, 0.0026, 0.0035,
                      0.0028, 0.0025, 0.0027, 0.0026, 0.0031, 0.0037, 0.0037, 0.0033, 0.0030,
                      0.0025, 0.0039, 0.0032, 0.0033, 0.0028, 0.0031, 0.0024, 0.0031, 0.0030,
                      0.0026, 0.0029, 0.0025, 0.0022, 0.0024, 0.0023, 0.0034, 0.0031, 0.0028,
                      0.0029, 0.0032, 0.0024, 0.0032, 0.0029, 0.0029, 0.0019, 0.0030, 0.0026,
                      0.0018, 0.0032, 0.0034, 0.0027, 0.0030, 0.0042, 0.0022, 0.0030, 0.0031,
                      0.0030, 0.0024, 0.0030, 0.0028, 0.0028, 0.0032, 0.0021, 0.0032, 0.0028,
                      0.0026, 0.0026, 0.0033, 0.0032, 0.0025, 0.0031, 0.0041, 0.0026, 0.0032,
                      0.0030, 0.0032, 0.0024, 0.0022, 0.0035, 0.0026, 0.0032, 0.0027, 0.0030,
                      0.0032, 0.0028, 0.0033, 0.0028, 0.0029, 0.0037, 0.0025, 0.0028, 0.0024,
                      0.0026, 0.0030, 0.0027, 0.0029, 0.0027, 0.0026, 0.0040, 0.0030, 0.0026,
                      0.0029, 0.0031, 0.0025, 0.0032, 0.0025, 0.0036, 0.0024, 0.0033, 0.0031,
                      0.0032, 0.0029, 0.0030, 0.0026, 0.0030, 0.0031, 0.0025, 0.0022, 0.0029,
                      0.0028, 0.0032]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.3638, -0.4012, -0.3694, -0.3866, -0.3468, -0.3392, -0.2638, -0.3324,
                        -0.4056, -0.3325, -0.3177, -0.3226, -0.3303, -0.3342, -0.4701, -0.3506,
                        -0.4197, -0.3849, -0.3153, -0.4996, -0.4094, -0.4172, -0.3589, -0.3984,
                        -0.2853, -0.3379, -0.3844, -0.3242, -0.3690, -0.3159, -0.2666, -0.2893,
                        -0.2607, -0.3862, -0.3275, -0.3562, -0.3694, -0.4064, -0.3089, -0.3662,
                        -0.3609, -0.3671, -0.1902, -0.2872, -0.3389, -0.1942, -0.4074, -0.3017,
                        -0.3490, -0.3486, -0.3820, -0.2324, -0.2787, -0.3947, -0.3792, -0.3045,
                        -0.3224, -0.3578, -0.3164, -0.4037, -0.2254, -0.4114, -0.3564, -0.3304,
                        -0.3283, -0.3468, -0.4094, -0.3156, -0.3977, -0.3879, -0.3273, -0.3636,
                        -0.3377, -0.3755, -0.3054, -0.2788, -0.3600, -0.3364, -0.3449, -0.3317,
                        -0.3832, -0.4153, -0.3528, -0.4166, -0.3541, -0.3746, -0.4786, -0.2787,
                        -0.3530, -0.3055, -0.2958, -0.3825, -0.3498, -0.3511, -0.3300, -0.2939,
                        -0.3585, -0.2892, -0.3312, -0.3534, -0.3976, -0.3245, -0.4120, -0.3087,
                        -0.3431, -0.3101, -0.4213, -0.3463, -0.4061, -0.3351, -0.3817, -0.3314,
                        -0.3882, -0.2946, -0.3030, -0.2779, -0.2964, -0.3199, -0.2961]), max_val=tensor([0.3406, 0.3677, 0.3266, 0.3504, 0.3593, 0.3087, 0.3459, 0.3246, 0.4481,
                        0.3589, 0.2730, 0.3486, 0.3328, 0.3924, 0.3789, 0.4752, 0.3915, 0.3631,
                        0.2429, 0.3485, 0.3724, 0.3646, 0.3371, 0.3913, 0.3059, 0.3947, 0.3671,
                        0.3308, 0.3197, 0.3150, 0.2824, 0.3063, 0.2883, 0.4308, 0.3986, 0.3560,
                        0.3517, 0.3285, 0.2716, 0.4087, 0.3734, 0.3734, 0.2362, 0.3778, 0.3047,
                        0.2254, 0.4021, 0.4358, 0.2932, 0.3869, 0.5319, 0.2746, 0.3843, 0.3607,
                        0.3465, 0.2986, 0.3828, 0.3315, 0.3517, 0.3407, 0.2651, 0.3829, 0.2989,
                        0.3144, 0.3266, 0.4207, 0.3185, 0.2797, 0.2903, 0.5167, 0.3190, 0.4112,
                        0.3818, 0.4062, 0.2895, 0.2508, 0.4390, 0.3299, 0.4102, 0.3401, 0.3309,
                        0.4049, 0.3415, 0.3418, 0.3246, 0.3639, 0.3708, 0.3163, 0.2991, 0.2785,
                        0.3277, 0.2306, 0.2872, 0.3738, 0.3492, 0.3298, 0.5128, 0.3794, 0.3277,
                        0.3720, 0.3484, 0.2878, 0.3247, 0.3175, 0.4603, 0.2983, 0.4045, 0.3979,
                        0.3649, 0.3703, 0.3670, 0.3124, 0.3575, 0.3977, 0.3164, 0.2781, 0.3692,
                        0.3617, 0.4013])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.7013]), zero_point=tensor([62], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-43.271636962890625, max_val=45.79351806640625)
            )
          )
          (bn2): BatchNorm2d(
            119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0329]), zero_point=tensor([61], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.9999687671661377, max_val=2.174226999282837)
            )
          )
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(
              55, 119, kernel_size=(1, 1), stride=(2, 2), bias=False
              (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0031, 0.0025, 0.0027, 0.0040, 0.0023, 0.0043, 0.0023, 0.0027, 0.0037,
                        0.0030, 0.0025, 0.0023, 0.0025, 0.0035, 0.0023, 0.0025, 0.0023, 0.0029,
                        0.0030, 0.0033, 0.0030, 0.0030, 0.0030, 0.0030, 0.0031, 0.0030, 0.0027,
                        0.0043, 0.0029, 0.0029, 0.0029, 0.0026, 0.0033, 0.0024, 0.0035, 0.0028,
                        0.0029, 0.0029, 0.0032, 0.0023, 0.0029, 0.0026, 0.0026, 0.0044, 0.0022,
                        0.0024, 0.0030, 0.0029, 0.0026, 0.0033, 0.0028, 0.0029, 0.0037, 0.0028,
                        0.0051, 0.0032, 0.0028, 0.0029, 0.0026, 0.0024, 0.0034, 0.0040, 0.0026,
                        0.0028, 0.0025, 0.0047, 0.0029, 0.0035, 0.0028, 0.0038, 0.0028, 0.0028,
                        0.0038, 0.0022, 0.0037, 0.0039, 0.0033, 0.0027, 0.0026, 0.0027, 0.0023,
                        0.0025, 0.0032, 0.0022, 0.0027, 0.0032, 0.0024, 0.0028, 0.0024, 0.0026,
                        0.0035, 0.0025, 0.0022, 0.0025, 0.0044, 0.0023, 0.0025, 0.0030, 0.0032,
                        0.0037, 0.0023, 0.0028, 0.0027, 0.0026, 0.0028, 0.0026, 0.0021, 0.0028,
                        0.0037, 0.0021, 0.0026, 0.0026, 0.0030, 0.0027, 0.0030, 0.0027, 0.0028,
                        0.0027, 0.0023]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                  min_val=tensor([-0.3972, -0.3174, -0.3503, -0.5080, -0.2950, -0.2366, -0.2895, -0.2228,
                          -0.4795, -0.3839, -0.3173, -0.2902, -0.3205, -0.2483, -0.2995, -0.3176,
                          -0.2309, -0.3612, -0.3847, -0.4260, -0.3804, -0.3790, -0.3333, -0.3455,
                          -0.3988, -0.3891, -0.2802, -0.5550, -0.3064, -0.3673, -0.3688, -0.3298,
                          -0.4263, -0.2910, -0.4456, -0.3602, -0.2396, -0.3733, -0.4107, -0.2973,
                          -0.3656, -0.3271, -0.3378, -0.5589, -0.2452, -0.3077, -0.3190, -0.3766,
                          -0.3285, -0.4213, -0.3411, -0.3698, -0.3435, -0.3522, -0.6556, -0.4085,
                          -0.3527, -0.3698, -0.3048, -0.3059, -0.3426, -0.1880, -0.3369, -0.3614,
                          -0.3160, -0.6012, -0.2961, -0.4492, -0.3511, -0.2564, -0.2521, -0.3605,
                          -0.4913, -0.2696, -0.4730, -0.3514, -0.3515, -0.3451, -0.3356, -0.2654,
                          -0.2910, -0.3138, -0.4117, -0.2800, -0.1643, -0.3553, -0.2871, -0.3575,
                          -0.3037, -0.1939, -0.4521, -0.3260, -0.2363, -0.3200, -0.5668, -0.2903,
                          -0.3166, -0.3015, -0.4137, -0.4687, -0.2651, -0.2913, -0.3511, -0.3316,
                          -0.3246, -0.3321, -0.2680, -0.3536, -0.4688, -0.2694, -0.3330, -0.3347,
                          -0.3817, -0.3274, -0.3806, -0.3494, -0.3525, -0.2834, -0.1413]), max_val=tensor([0.2486, 0.2451, 0.1942, 0.3046, 0.2341, 0.5514, 0.2447, 0.3375, 0.2125,
                          0.2178, 0.2844, 0.2919, 0.2433, 0.4467, 0.2846, 0.3194, 0.2960, 0.3714,
                          0.2926, 0.2458, 0.2945, 0.3340, 0.3824, 0.3818, 0.2910, 0.2791, 0.3455,
                          0.2506, 0.3734, 0.3485, 0.1970, 0.2255, 0.2707, 0.3013, 0.3580, 0.2335,
                          0.3735, 0.3283, 0.3163, 0.2435, 0.2036, 0.3077, 0.3200, 0.2279, 0.2845,
                          0.2817, 0.3791, 0.2994, 0.3081, 0.2585, 0.3614, 0.2655, 0.4699, 0.2691,
                          0.2580, 0.2190, 0.2912, 0.2593, 0.3312, 0.2700, 0.4271, 0.5094, 0.3141,
                          0.2776, 0.2581, 0.3039, 0.3729, 0.3624, 0.3500, 0.4868, 0.3616, 0.2597,
                          0.3009, 0.2798, 0.2896, 0.5006, 0.4226, 0.2479, 0.2795, 0.3376, 0.2454,
                          0.3141, 0.2964, 0.2714, 0.3442, 0.4077, 0.3103, 0.2991, 0.2774, 0.3320,
                          0.3210, 0.2001, 0.2849, 0.2487, 0.2481, 0.2247, 0.2620, 0.3853, 0.2613,
                          0.2895, 0.2963, 0.3578, 0.2436, 0.2904, 0.3527, 0.2828, 0.2620, 0.3523,
                          0.2324, 0.2695, 0.3005, 0.3215, 0.3302, 0.3390, 0.2702, 0.2935, 0.3599,
                          0.3419, 0.2969])
                )
              )
              (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1455]), zero_point=tensor([75], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=-10.91691780090332, max_val=7.563588619232178)
              )
            )
            (1): BatchNorm2d(
              119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1052]), zero_point=tensor([65], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=-6.854622840881348, max_val=6.509270191192627)
              )
            )
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(
            119, 125, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0033, 0.0024, 0.0028, 0.0025, 0.0029, 0.0024, 0.0025, 0.0029, 0.0028,
                      0.0029, 0.0022, 0.0030, 0.0029, 0.0028, 0.0027, 0.0029, 0.0031, 0.0029,
                      0.0025, 0.0039, 0.0031, 0.0022, 0.0027, 0.0023, 0.0023, 0.0030, 0.0030,
                      0.0027, 0.0034, 0.0028, 0.0021, 0.0026, 0.0031, 0.0023, 0.0027, 0.0024,
                      0.0024, 0.0032, 0.0032, 0.0029, 0.0024, 0.0030, 0.0025, 0.0024, 0.0031,
                      0.0028, 0.0023, 0.0029, 0.0031, 0.0028, 0.0019, 0.0024, 0.0030, 0.0021,
                      0.0026, 0.0024, 0.0024, 0.0022, 0.0024, 0.0023, 0.0028, 0.0030, 0.0026,
                      0.0021, 0.0027, 0.0039, 0.0031, 0.0027, 0.0027, 0.0027, 0.0028, 0.0026,
                      0.0022, 0.0026, 0.0028, 0.0026, 0.0023, 0.0025, 0.0028, 0.0031, 0.0023,
                      0.0029, 0.0025, 0.0029, 0.0031, 0.0030, 0.0023, 0.0038, 0.0027, 0.0023,
                      0.0023, 0.0027, 0.0025, 0.0028, 0.0028, 0.0027, 0.0024, 0.0024, 0.0030,
                      0.0029, 0.0028, 0.0025, 0.0029, 0.0029, 0.0026, 0.0023, 0.0024, 0.0029,
                      0.0024, 0.0034, 0.0026, 0.0025, 0.0021, 0.0022, 0.0030, 0.0028, 0.0027,
                      0.0021, 0.0028, 0.0027, 0.0029, 0.0024, 0.0024, 0.0027, 0.0024]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.4207, -0.2557, -0.3092, -0.3261, -0.3650, -0.3107, -0.3187, -0.3453,
                        -0.3576, -0.3741, -0.2815, -0.3827, -0.3670, -0.3622, -0.3394, -0.3681,
                        -0.3961, -0.3669, -0.3235, -0.3729, -0.3943, -0.2814, -0.3519, -0.2900,
                        -0.2989, -0.3799, -0.3875, -0.3515, -0.4302, -0.3648, -0.2733, -0.3267,
                        -0.3921, -0.2876, -0.3446, -0.3056, -0.3057, -0.4044, -0.4144, -0.3736,
                        -0.3021, -0.3853, -0.2586, -0.3024, -0.3593, -0.3324, -0.2944, -0.3770,
                        -0.4024, -0.3643, -0.2432, -0.3044, -0.3192, -0.2675, -0.3391, -0.3069,
                        -0.3070, -0.2855, -0.3116, -0.2947, -0.3538, -0.3787, -0.3379, -0.2677,
                        -0.3295, -0.4945, -0.4029, -0.3459, -0.3505, -0.3423, -0.3552, -0.3373,
                        -0.2844, -0.3375, -0.3601, -0.3384, -0.2919, -0.3178, -0.3054, -0.4007,
                        -0.2974, -0.3754, -0.3258, -0.3754, -0.4014, -0.3870, -0.2945, -0.4897,
                        -0.3403, -0.2931, -0.2958, -0.2960, -0.3189, -0.3573, -0.3561, -0.3407,
                        -0.3012, -0.3106, -0.3851, -0.3703, -0.3609, -0.3217, -0.3660, -0.3669,
                        -0.2973, -0.2936, -0.3063, -0.3751, -0.2907, -0.4375, -0.3302, -0.3210,
                        -0.2657, -0.2492, -0.3781, -0.3565, -0.3459, -0.2740, -0.3548, -0.3418,
                        -0.3683, -0.3135, -0.3100, -0.3437, -0.3027]), max_val=tensor([0.3114, 0.3092, 0.3571, 0.2665, 0.2665, 0.2742, 0.2994, 0.3679, 0.3227,
                        0.3189, 0.2024, 0.2573, 0.3151, 0.3016, 0.2747, 0.2945, 0.2593, 0.3308,
                        0.2917, 0.4921, 0.2663, 0.2523, 0.2989, 0.2787, 0.2471, 0.2458, 0.2572,
                        0.2590, 0.2983, 0.2390, 0.2227, 0.2664, 0.3019, 0.2884, 0.2900, 0.3038,
                        0.2437, 0.3020, 0.2742, 0.2945, 0.2368, 0.3129, 0.3118, 0.2315, 0.3928,
                        0.3502, 0.2920, 0.2341, 0.3626, 0.2898, 0.2220, 0.2855, 0.3835, 0.2400,
                        0.2908, 0.2757, 0.2809, 0.2264, 0.2849, 0.2197, 0.2451, 0.2943, 0.2731,
                        0.2425, 0.3396, 0.2995, 0.2800, 0.2939, 0.3161, 0.2647, 0.3129, 0.2301,
                        0.2434, 0.2861, 0.2536, 0.3230, 0.2634, 0.2903, 0.3497, 0.2617, 0.2173,
                        0.2287, 0.2647, 0.3228, 0.2573, 0.3157, 0.2664, 0.2746, 0.3411, 0.2526,
                        0.2439, 0.3445, 0.2507, 0.3447, 0.3102, 0.2536, 0.2158, 0.2872, 0.2485,
                        0.3114, 0.2274, 0.2582, 0.2291, 0.2459, 0.3267, 0.2157, 0.3028, 0.2858,
                        0.3011, 0.2556, 0.2888, 0.2492, 0.2026, 0.2750, 0.2979, 0.2595, 0.2574,
                        0.2315, 0.3035, 0.2951, 0.3035, 0.2420, 0.2546, 0.2950, 0.2506])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.4224]), zero_point=tensor([91], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-38.4554443359375, max_val=15.188919067382812)
            )
          )
          (bn1): BatchNorm2d(
            125, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1018]), zero_point=tensor([76], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-7.778568267822266, max_val=5.154354095458984)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            125, 119, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0016, 0.0025, 0.0027, 0.0028, 0.0031, 0.0024, 0.0020, 0.0027, 0.0025,
                      0.0027, 0.0027, 0.0028, 0.0028, 0.0024, 0.0031, 0.0032, 0.0017, 0.0028,
                      0.0025, 0.0030, 0.0028, 0.0034, 0.0028, 0.0029, 0.0019, 0.0027, 0.0021,
                      0.0023, 0.0023, 0.0030, 0.0024, 0.0025, 0.0023, 0.0029, 0.0021, 0.0026,
                      0.0033, 0.0030, 0.0028, 0.0031, 0.0025, 0.0030, 0.0017, 0.0029, 0.0030,
                      0.0021, 0.0032, 0.0027, 0.0023, 0.0025, 0.0033, 0.0025, 0.0019, 0.0028,
                      0.0026, 0.0025, 0.0030, 0.0025, 0.0025, 0.0030, 0.0039, 0.0025, 0.0026,
                      0.0025, 0.0023, 0.0030, 0.0022, 0.0025, 0.0027, 0.0033, 0.0025, 0.0029,
                      0.0019, 0.0028, 0.0028, 0.0021, 0.0026, 0.0025, 0.0026, 0.0024, 0.0026,
                      0.0027, 0.0026, 0.0024, 0.0021, 0.0029, 0.0031, 0.0023, 0.0024, 0.0023,
                      0.0022, 0.0026, 0.0025, 0.0030, 0.0024, 0.0030, 0.0028, 0.0023, 0.0027,
                      0.0034, 0.0028, 0.0037, 0.0019, 0.0027, 0.0026, 0.0023, 0.0026, 0.0026,
                      0.0024, 0.0027, 0.0025, 0.0024, 0.0029, 0.0030, 0.0025, 0.0028, 0.0028,
                      0.0024, 0.0027]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2086, -0.3250, -0.3455, -0.3121, -0.3071, -0.2989, -0.2598, -0.3471,
                        -0.3244, -0.2978, -0.3441, -0.3145, -0.3647, -0.3080, -0.3982, -0.3714,
                        -0.2153, -0.3642, -0.3261, -0.3843, -0.3608, -0.4302, -0.3366, -0.3495,
                        -0.2410, -0.3479, -0.2533, -0.2213, -0.2922, -0.3903, -0.3013, -0.3156,
                        -0.2960, -0.3258, -0.2636, -0.2801, -0.4172, -0.3387, -0.2537, -0.4029,
                        -0.3249, -0.3845, -0.2232, -0.3714, -0.3815, -0.2331, -0.3225, -0.2666,
                        -0.2946, -0.2989, -0.3725, -0.3150, -0.2398, -0.2708, -0.2486, -0.3155,
                        -0.3794, -0.2911, -0.2601, -0.2944, -0.5024, -0.2836, -0.2895, -0.2703,
                        -0.2785, -0.3829, -0.2646, -0.3212, -0.3485, -0.4200, -0.3235, -0.3736,
                        -0.2440, -0.3592, -0.3601, -0.2561, -0.3383, -0.3173, -0.2839, -0.2815,
                        -0.3346, -0.3479, -0.3371, -0.3036, -0.2728, -0.3371, -0.2966, -0.2954,
                        -0.2701, -0.2929, -0.2871, -0.2965, -0.3067, -0.3892, -0.2926, -0.2959,
                        -0.3609, -0.2895, -0.3450, -0.4312, -0.3559, -0.4727, -0.2098, -0.2827,
                        -0.3376, -0.2978, -0.3364, -0.3308, -0.2590, -0.3379, -0.2656, -0.2956,
                        -0.3676, -0.2848, -0.2801, -0.3642, -0.3615, -0.3134, -0.3400]), max_val=tensor([0.2077, 0.3149, 0.3004, 0.3537, 0.3882, 0.3075, 0.2228, 0.3425, 0.2966,
                        0.3379, 0.2796, 0.3494, 0.3439, 0.2256, 0.3282, 0.4056, 0.1974, 0.3095,
                        0.2793, 0.2444, 0.3154, 0.3576, 0.3547, 0.3632, 0.2346, 0.2837, 0.2669,
                        0.2977, 0.2884, 0.3043, 0.2145, 0.2764, 0.2911, 0.3743, 0.2571, 0.3271,
                        0.2996, 0.3839, 0.3549, 0.3121, 0.2615, 0.3594, 0.2078, 0.3000, 0.3637,
                        0.2631, 0.4046, 0.3417, 0.2591, 0.3215, 0.4183, 0.3203, 0.2467, 0.3612,
                        0.3250, 0.3007, 0.2684, 0.3225, 0.3231, 0.3777, 0.2585, 0.3153, 0.3352,
                        0.3115, 0.2871, 0.3484, 0.2804, 0.3080, 0.3086, 0.4078, 0.2499, 0.3679,
                        0.2462, 0.2667, 0.3090, 0.2642, 0.2932, 0.2899, 0.3313, 0.3041, 0.2918,
                        0.3099, 0.2826, 0.3073, 0.2691, 0.3734, 0.3902, 0.2476, 0.3052, 0.2805,
                        0.2603, 0.3341, 0.3209, 0.2859, 0.3048, 0.3814, 0.3160, 0.2962, 0.2429,
                        0.3466, 0.3031, 0.3178, 0.2376, 0.3402, 0.2943, 0.2961, 0.3167, 0.3350,
                        0.3026, 0.3370, 0.3209, 0.3004, 0.2859, 0.3798, 0.3125, 0.2681, 0.3171,
                        0.2895, 0.3408])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.3812]), zero_point=tensor([64], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-24.383649826049805, max_val=24.024715423583984)
            )
          )
          (bn2): BatchNorm2d(
            119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0213]), zero_point=tensor([69], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.4754410982131958, max_val=1.2357683181762695)
            )
          )
          (act2): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(
            119, 253, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0026, 0.0024, 0.0028, 0.0028, 0.0024, 0.0024, 0.0030, 0.0027, 0.0025,
                      0.0024, 0.0025, 0.0026, 0.0024, 0.0027, 0.0028, 0.0024, 0.0032, 0.0025,
                      0.0025, 0.0026, 0.0025, 0.0029, 0.0027, 0.0029, 0.0029, 0.0030, 0.0022,
                      0.0033, 0.0031, 0.0027, 0.0023, 0.0024, 0.0021, 0.0026, 0.0026, 0.0031,
                      0.0022, 0.0029, 0.0031, 0.0030, 0.0024, 0.0028, 0.0025, 0.0030, 0.0029,
                      0.0025, 0.0032, 0.0022, 0.0032, 0.0028, 0.0026, 0.0028, 0.0030, 0.0028,
                      0.0026, 0.0029, 0.0031, 0.0021, 0.0027, 0.0027, 0.0027, 0.0025, 0.0023,
                      0.0025, 0.0024, 0.0029, 0.0022, 0.0024, 0.0026, 0.0028, 0.0029, 0.0027,
                      0.0028, 0.0027, 0.0027, 0.0029, 0.0024, 0.0031, 0.0029, 0.0026, 0.0029,
                      0.0023, 0.0028, 0.0025, 0.0024, 0.0020, 0.0032, 0.0027, 0.0026, 0.0026,
                      0.0027, 0.0029, 0.0029, 0.0023, 0.0025, 0.0022, 0.0027, 0.0023, 0.0026,
                      0.0024, 0.0028, 0.0033, 0.0027, 0.0027, 0.0018, 0.0024, 0.0026, 0.0024,
                      0.0028, 0.0026, 0.0028, 0.0025, 0.0027, 0.0028, 0.0026, 0.0023, 0.0024,
                      0.0028, 0.0029, 0.0033, 0.0022, 0.0026, 0.0027, 0.0031, 0.0026, 0.0022,
                      0.0023, 0.0024, 0.0027, 0.0023, 0.0028, 0.0028, 0.0021, 0.0026, 0.0032,
                      0.0032, 0.0025, 0.0030, 0.0029, 0.0036, 0.0029, 0.0024, 0.0025, 0.0031,
                      0.0021, 0.0024, 0.0030, 0.0025, 0.0025, 0.0025, 0.0020, 0.0028, 0.0031,
                      0.0027, 0.0029, 0.0028, 0.0025, 0.0025, 0.0022, 0.0027, 0.0027, 0.0025,
                      0.0029, 0.0033, 0.0023, 0.0025, 0.0024, 0.0033, 0.0032, 0.0025, 0.0026,
                      0.0024, 0.0029, 0.0032, 0.0030, 0.0027, 0.0024, 0.0030, 0.0021, 0.0027,
                      0.0035, 0.0023, 0.0021, 0.0034, 0.0026, 0.0031, 0.0027, 0.0031, 0.0026,
                      0.0020, 0.0025, 0.0029, 0.0020, 0.0026, 0.0026, 0.0021, 0.0029, 0.0030,
                      0.0028, 0.0025, 0.0030, 0.0028, 0.0025, 0.0028, 0.0025, 0.0028, 0.0028,
                      0.0023, 0.0022, 0.0031, 0.0028, 0.0023, 0.0022, 0.0027, 0.0022, 0.0024,
                      0.0024, 0.0023, 0.0034, 0.0027, 0.0024, 0.0027, 0.0026, 0.0028, 0.0027,
                      0.0033, 0.0023, 0.0027, 0.0024, 0.0024, 0.0019, 0.0029, 0.0026, 0.0027,
                      0.0025, 0.0028, 0.0027, 0.0036, 0.0030, 0.0023, 0.0026, 0.0031, 0.0027,
                      0.0022, 0.0027, 0.0026, 0.0025, 0.0029, 0.0024, 0.0032, 0.0022, 0.0031,
                      0.0026]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.3364, -0.2932, -0.3556, -0.3587, -0.2980, -0.3024, -0.3796, -0.3500,
                        -0.3138, -0.3089, -0.3235, -0.3270, -0.3010, -0.3414, -0.3618, -0.3022,
                        -0.3068, -0.3144, -0.3203, -0.3268, -0.3153, -0.3666, -0.3502, -0.3690,
                        -0.3772, -0.3839, -0.2832, -0.4226, -0.3988, -0.3488, -0.3008, -0.3134,
                        -0.2415, -0.3348, -0.3305, -0.3663, -0.2854, -0.3774, -0.3941, -0.3882,
                        -0.3049, -0.3617, -0.3203, -0.3795, -0.3767, -0.3157, -0.4130, -0.2863,
                        -0.4131, -0.3524, -0.2875, -0.3540, -0.3790, -0.3552, -0.3188, -0.3399,
                        -0.3991, -0.2737, -0.3409, -0.2932, -0.3217, -0.2519, -0.2886, -0.3170,
                        -0.3040, -0.3658, -0.2592, -0.3019, -0.3011, -0.3555, -0.3722, -0.3496,
                        -0.3571, -0.3363, -0.3455, -0.3772, -0.3075, -0.3974, -0.3736, -0.3316,
                        -0.3758, -0.2984, -0.3634, -0.3199, -0.3087, -0.2544, -0.4067, -0.3477,
                        -0.3276, -0.3329, -0.3444, -0.2376, -0.3751, -0.2937, -0.3146, -0.2853,
                        -0.2581, -0.2994, -0.3370, -0.3084, -0.3575, -0.4273, -0.3400, -0.3403,
                        -0.2156, -0.2607, -0.3271, -0.3136, -0.3627, -0.3356, -0.3543, -0.3201,
                        -0.3472, -0.3562, -0.3318, -0.2750, -0.3098, -0.3634, -0.3664, -0.4237,
                        -0.2678, -0.3328, -0.3421, -0.3928, -0.3351, -0.2855, -0.2979, -0.3070,
                        -0.3461, -0.2903, -0.3627, -0.3526, -0.2696, -0.3323, -0.4103, -0.4093,
                        -0.3195, -0.3800, -0.3694, -0.4567, -0.3651, -0.3114, -0.3243, -0.3978,
                        -0.2746, -0.3034, -0.3821, -0.3231, -0.3244, -0.2950, -0.2407, -0.3290,
                        -0.3970, -0.3394, -0.3703, -0.3543, -0.3182, -0.3259, -0.2835, -0.3410,
                        -0.3398, -0.3241, -0.3775, -0.4204, -0.2899, -0.3147, -0.3068, -0.4179,
                        -0.4041, -0.3264, -0.3278, -0.2653, -0.3723, -0.4092, -0.3897, -0.3492,
                        -0.3009, -0.3785, -0.2625, -0.3430, -0.4542, -0.2641, -0.2659, -0.4297,
                        -0.3298, -0.3967, -0.3487, -0.3964, -0.3371, -0.2498, -0.3247, -0.3776,
                        -0.2558, -0.3289, -0.3321, -0.2640, -0.3429, -0.3885, -0.3632, -0.3187,
                        -0.3842, -0.3547, -0.2942, -0.3595, -0.3216, -0.3534, -0.3611, -0.2955,
                        -0.2809, -0.3911, -0.3581, -0.2883, -0.2857, -0.3396, -0.2749, -0.3024,
                        -0.3068, -0.2630, -0.4408, -0.3470, -0.2898, -0.2992, -0.3357, -0.3287,
                        -0.3436, -0.4208, -0.2903, -0.3457, -0.3103, -0.2784, -0.2480, -0.3708,
                        -0.3389, -0.3414, -0.3163, -0.3623, -0.3446, -0.4669, -0.3776, -0.2885,
                        -0.3358, -0.4031, -0.3494, -0.2759, -0.3423, -0.3133, -0.3223, -0.3299,
                        -0.3109, -0.4036, -0.2801, -0.4017, -0.3373]), max_val=tensor([0.2402, 0.3039, 0.2866, 0.2583, 0.3064, 0.2687, 0.3076, 0.2976, 0.2868,
                        0.2906, 0.2979, 0.2743, 0.2800, 0.2792, 0.2581, 0.2547, 0.4038, 0.2945,
                        0.3064, 0.3037, 0.2935, 0.2428, 0.2784, 0.2674, 0.2014, 0.2836, 0.2680,
                        0.3586, 0.2604, 0.2842, 0.2572, 0.2465, 0.2666, 0.2803, 0.2753, 0.3917,
                        0.2804, 0.3129, 0.2847, 0.2449, 0.2942, 0.2478, 0.2668, 0.3346, 0.2472,
                        0.3214, 0.3675, 0.2547, 0.3680, 0.2614, 0.3286, 0.2331, 0.3200, 0.2660,
                        0.3290, 0.3673, 0.2623, 0.2687, 0.3268, 0.3404, 0.3467, 0.3132, 0.2072,
                        0.2640, 0.2977, 0.3132, 0.2841, 0.2135, 0.3264, 0.2399, 0.2529, 0.2290,
                        0.2020, 0.3406, 0.3122, 0.2557, 0.2818, 0.2759, 0.2693, 0.2550, 0.2771,
                        0.2405, 0.2899, 0.2936, 0.2868, 0.2579, 0.2851, 0.3198, 0.2734, 0.2263,
                        0.2618, 0.3677, 0.2709, 0.2700, 0.3184, 0.2199, 0.3458, 0.1835, 0.2684,
                        0.2538, 0.3065, 0.2078, 0.2760, 0.3024, 0.2324, 0.3051, 0.2517, 0.2176,
                        0.2651, 0.3034, 0.2464, 0.2299, 0.3091, 0.2353, 0.2351, 0.2930, 0.2409,
                        0.2792, 0.2981, 0.2473, 0.2799, 0.2433, 0.2681, 0.2609, 0.2338, 0.2323,
                        0.2649, 0.2587, 0.3183, 0.2893, 0.2245, 0.3143, 0.2401, 0.2581, 0.2297,
                        0.3080, 0.2373, 0.2564, 0.2854, 0.3051, 0.2858, 0.2605, 0.2446, 0.2800,
                        0.2664, 0.2849, 0.3277, 0.2637, 0.2768, 0.3144, 0.2492, 0.3527, 0.2723,
                        0.2827, 0.2961, 0.2876, 0.2443, 0.2790, 0.2664, 0.2109, 0.2952, 0.1989,
                        0.2983, 0.2418, 0.2599, 0.2842, 0.2618, 0.2300, 0.3073, 0.2440, 0.2174,
                        0.3085, 0.2492, 0.2429, 0.2827, 0.3255, 0.2830, 0.2790, 0.2333, 0.3193,
                        0.3541, 0.2888, 0.2677, 0.3050, 0.1866, 0.2635, 0.2933, 0.3863, 0.2776,
                        0.2469, 0.2999, 0.2689, 0.2220, 0.2587, 0.2890, 0.2151, 0.3689, 0.3227,
                        0.2510, 0.2778, 0.2847, 0.2631, 0.3237, 0.2654, 0.2909, 0.2462, 0.2150,
                        0.2310, 0.2313, 0.3025, 0.2828, 0.2929, 0.2074, 0.2357, 0.2761, 0.2990,
                        0.2673, 0.2929, 0.3214, 0.2922, 0.3001, 0.3465, 0.2919, 0.3553, 0.2338,
                        0.2667, 0.2551, 0.2640, 0.2859, 0.3039, 0.2444, 0.2586, 0.2369, 0.2749,
                        0.2590, 0.2788, 0.2347, 0.3591, 0.2521, 0.2440, 0.2353, 0.2724, 0.2514,
                        0.2073, 0.2838, 0.3248, 0.2901, 0.3623, 0.2910, 0.3020, 0.2676, 0.3528,
                        0.2570])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.4426]), zero_point=tensor([79], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-34.996551513671875, max_val=21.214824676513672)
            )
          )
          (bn1): BatchNorm2d(
            253, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1108]), zero_point=tensor([68], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-7.55365514755249, max_val=6.514064311981201)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            253, 247, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0023, 0.0023, 0.0019, 0.0028, 0.0031, 0.0031, 0.0029, 0.0026, 0.0027,
                      0.0031, 0.0027, 0.0031, 0.0028, 0.0025, 0.0030, 0.0026, 0.0031, 0.0025,
                      0.0030, 0.0024, 0.0024, 0.0029, 0.0029, 0.0023, 0.0024, 0.0032, 0.0026,
                      0.0030, 0.0028, 0.0029, 0.0025, 0.0026, 0.0022, 0.0034, 0.0029, 0.0030,
                      0.0025, 0.0025, 0.0030, 0.0025, 0.0030, 0.0028, 0.0033, 0.0026, 0.0030,
                      0.0032, 0.0029, 0.0026, 0.0025, 0.0023, 0.0022, 0.0024, 0.0022, 0.0024,
                      0.0026, 0.0028, 0.0031, 0.0030, 0.0030, 0.0025, 0.0026, 0.0030, 0.0023,
                      0.0024, 0.0025, 0.0029, 0.0024, 0.0023, 0.0034, 0.0030, 0.0023, 0.0032,
                      0.0022, 0.0027, 0.0031, 0.0028, 0.0031, 0.0028, 0.0026, 0.0025, 0.0028,
                      0.0030, 0.0025, 0.0030, 0.0022, 0.0027, 0.0025, 0.0028, 0.0026, 0.0023,
                      0.0018, 0.0023, 0.0026, 0.0026, 0.0029, 0.0024, 0.0027, 0.0027, 0.0025,
                      0.0030, 0.0027, 0.0023, 0.0025, 0.0027, 0.0027, 0.0023, 0.0031, 0.0024,
                      0.0025, 0.0024, 0.0026, 0.0022, 0.0028, 0.0028, 0.0031, 0.0027, 0.0025,
                      0.0028, 0.0025, 0.0024, 0.0025, 0.0020, 0.0027, 0.0028, 0.0034, 0.0027,
                      0.0025, 0.0027, 0.0029, 0.0030, 0.0034, 0.0027, 0.0026, 0.0027, 0.0034,
                      0.0028, 0.0031, 0.0028, 0.0025, 0.0027, 0.0025, 0.0026, 0.0028, 0.0034,
                      0.0025, 0.0026, 0.0025, 0.0026, 0.0020, 0.0029, 0.0022, 0.0026, 0.0024,
                      0.0032, 0.0024, 0.0028, 0.0026, 0.0020, 0.0032, 0.0026, 0.0023, 0.0025,
                      0.0023, 0.0029, 0.0030, 0.0029, 0.0027, 0.0023, 0.0023, 0.0028, 0.0033,
                      0.0025, 0.0031, 0.0023, 0.0021, 0.0029, 0.0026, 0.0027, 0.0034, 0.0023,
                      0.0022, 0.0028, 0.0025, 0.0032, 0.0034, 0.0025, 0.0027, 0.0033, 0.0025,
                      0.0033, 0.0025, 0.0027, 0.0027, 0.0026, 0.0027, 0.0023, 0.0024, 0.0021,
                      0.0024, 0.0028, 0.0029, 0.0022, 0.0023, 0.0017, 0.0024, 0.0029, 0.0029,
                      0.0037, 0.0021, 0.0023, 0.0027, 0.0024, 0.0024, 0.0024, 0.0027, 0.0022,
                      0.0024, 0.0025, 0.0025, 0.0028, 0.0024, 0.0027, 0.0021, 0.0024, 0.0026,
                      0.0023, 0.0025, 0.0026, 0.0025, 0.0029, 0.0031, 0.0027, 0.0036, 0.0028,
                      0.0022, 0.0030, 0.0032, 0.0027, 0.0025, 0.0025, 0.0028, 0.0028, 0.0029,
                      0.0025, 0.0025, 0.0032, 0.0029]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2969, -0.2923, -0.2149, -0.3441, -0.3927, -0.3966, -0.3749, -0.3326,
                        -0.3400, -0.3554, -0.3020, -0.3924, -0.3260, -0.3166, -0.3483, -0.2934,
                        -0.3046, -0.2600, -0.3819, -0.3080, -0.3097, -0.3719, -0.3708, -0.2962,
                        -0.3025, -0.4108, -0.2935, -0.3189, -0.3559, -0.3493, -0.3206, -0.2679,
                        -0.2813, -0.3407, -0.2823, -0.3257, -0.2801, -0.3240, -0.2931, -0.3159,
                        -0.2654, -0.3553, -0.3938, -0.2951, -0.3046, -0.4146, -0.3221, -0.3183,
                        -0.2886, -0.2591, -0.2712, -0.2670, -0.2792, -0.2609, -0.2062, -0.2820,
                        -0.3141, -0.3805, -0.3821, -0.2983, -0.3357, -0.3849, -0.2963, -0.3085,
                        -0.3215, -0.3420, -0.2502, -0.3004, -0.4318, -0.3832, -0.2910, -0.3186,
                        -0.2852, -0.2681, -0.3616, -0.3594, -0.3917, -0.3526, -0.2930, -0.3165,
                        -0.3068, -0.2837, -0.2814, -0.3805, -0.2777, -0.3491, -0.3250, -0.2971,
                        -0.3387, -0.2881, -0.1841, -0.2979, -0.3128, -0.3022, -0.3766, -0.3074,
                        -0.2936, -0.3233, -0.3092, -0.3539, -0.3223, -0.2872, -0.3218, -0.3057,
                        -0.2488, -0.2641, -0.4008, -0.3036, -0.2644, -0.2782, -0.3305, -0.2814,
                        -0.2974, -0.3550, -0.4030, -0.3426, -0.2878, -0.2699, -0.3187, -0.3047,
                        -0.2766, -0.2463, -0.2554, -0.3579, -0.3737, -0.3444, -0.3174, -0.3215,
                        -0.3486, -0.3009, -0.4413, -0.3351, -0.2631, -0.3439, -0.2656, -0.3547,
                        -0.3906, -0.3618, -0.3245, -0.3473, -0.2708, -0.3357, -0.3563, -0.2928,
                        -0.3064, -0.3296, -0.2890, -0.3095, -0.2541, -0.3135, -0.2835, -0.3144,
                        -0.2824, -0.3265, -0.2485, -0.3620, -0.3304, -0.2404, -0.2945, -0.3297,
                        -0.2947, -0.2679, -0.2978, -0.2838, -0.3896, -0.3717, -0.3479, -0.2996,
                        -0.2909, -0.3124, -0.4226, -0.2879, -0.3231, -0.2957, -0.2665, -0.3704,
                        -0.3193, -0.2718, -0.3149, -0.2916, -0.2766, -0.2943, -0.3237, -0.3414,
                        -0.4306, -0.3138, -0.3458, -0.4264, -0.3150, -0.4177, -0.2915, -0.3469,
                        -0.2650, -0.2825, -0.2879, -0.2744, -0.2513, -0.2541, -0.3084, -0.3265,
                        -0.3733, -0.2798, -0.3001, -0.2119, -0.3029, -0.3175, -0.3215, -0.3786,
                        -0.2691, -0.2865, -0.3492, -0.3099, -0.3041, -0.3089, -0.2461, -0.2833,
                        -0.3089, -0.3248, -0.3109, -0.3136, -0.3017, -0.3412, -0.2734, -0.3120,
                        -0.3368, -0.2989, -0.3217, -0.2986, -0.2917, -0.3629, -0.2975, -0.2933,
                        -0.2700, -0.3619, -0.2878, -0.3903, -0.3898, -0.3309, -0.2554, -0.2696,
                        -0.3587, -0.3642, -0.3011, -0.2723, -0.3017, -0.2710, -0.3721]), max_val=tensor([0.2942, 0.2387, 0.2359, 0.3555, 0.2573, 0.3042, 0.3236, 0.2817, 0.2835,
                        0.3932, 0.3417, 0.2873, 0.3549, 0.3221, 0.3797, 0.3306, 0.3888, 0.3149,
                        0.3623, 0.2895, 0.3000, 0.2742, 0.3426, 0.2984, 0.2727, 0.3739, 0.3273,
                        0.3748, 0.3079, 0.3651, 0.3063, 0.3353, 0.2791, 0.4282, 0.3714, 0.3873,
                        0.3207, 0.2985, 0.3761, 0.3164, 0.3757, 0.2956, 0.4135, 0.3282, 0.3783,
                        0.2485, 0.3670, 0.3329, 0.3124, 0.2981, 0.2798, 0.3041, 0.2855, 0.3022,
                        0.3348, 0.3584, 0.3992, 0.2891, 0.3822, 0.3229, 0.3222, 0.2760, 0.2773,
                        0.3103, 0.3235, 0.3658, 0.3058, 0.2922, 0.3553, 0.2794, 0.2812, 0.4001,
                        0.2833, 0.3451, 0.3914, 0.2508, 0.3515, 0.3354, 0.3253, 0.3087, 0.3508,
                        0.3850, 0.3158, 0.2873, 0.2785, 0.3485, 0.2892, 0.3564, 0.2436, 0.2730,
                        0.2339, 0.2895, 0.3247, 0.3348, 0.2989, 0.2621, 0.3432, 0.3383, 0.3221,
                        0.3846, 0.3386, 0.2923, 0.3100, 0.3488, 0.3388, 0.2880, 0.3382, 0.3079,
                        0.3118, 0.3039, 0.3100, 0.2742, 0.3510, 0.3402, 0.2955, 0.2928, 0.3118,
                        0.3611, 0.2993, 0.2753, 0.3177, 0.2534, 0.3383, 0.3284, 0.4297, 0.3008,
                        0.2689, 0.3383, 0.3665, 0.3790, 0.3385, 0.3421, 0.3292, 0.3154, 0.4296,
                        0.2831, 0.3568, 0.2938, 0.3082, 0.2722, 0.3216, 0.2740, 0.3619, 0.4356,
                        0.3113, 0.3121, 0.3160, 0.3267, 0.2468, 0.3655, 0.2695, 0.3311, 0.3036,
                        0.4078, 0.3044, 0.3398, 0.3041, 0.2496, 0.4065, 0.3160, 0.2585, 0.3169,
                        0.2975, 0.3663, 0.2619, 0.3291, 0.3017, 0.2696, 0.2844, 0.3610, 0.3925,
                        0.3115, 0.3971, 0.2949, 0.2637, 0.3534, 0.3351, 0.3486, 0.4288, 0.2973,
                        0.2538, 0.3549, 0.2950, 0.4003, 0.2650, 0.3075, 0.3375, 0.2956, 0.2562,
                        0.2991, 0.3205, 0.2670, 0.3385, 0.3323, 0.3387, 0.2959, 0.2994, 0.2719,
                        0.2794, 0.3619, 0.3512, 0.2755, 0.2822, 0.2170, 0.2495, 0.3653, 0.3663,
                        0.4666, 0.2587, 0.2936, 0.2822, 0.2569, 0.2782, 0.3042, 0.3427, 0.2446,
                        0.2631, 0.3041, 0.3147, 0.3544, 0.2932, 0.3131, 0.2513, 0.2430, 0.3220,
                        0.2541, 0.3029, 0.3300, 0.3118, 0.3677, 0.3971, 0.3401, 0.4580, 0.3054,
                        0.2618, 0.2893, 0.4050, 0.3379, 0.3123, 0.3175, 0.2936, 0.3151, 0.3710,
                        0.3209, 0.3208, 0.4122, 0.2798])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.5098]), zero_point=tensor([67], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-34.04077911376953, max_val=30.70110321044922)
            )
          )
          (bn2): BatchNorm2d(
            247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0323]), zero_point=tensor([62], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.9993891716003418, max_val=2.097520351409912)
            )
          )
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(
              119, 247, kernel_size=(1, 1), stride=(2, 2), bias=False
              (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0021, 0.0018, 0.0020, 0.0022, 0.0027, 0.0021, 0.0032, 0.0021, 0.0026,
                        0.0022, 0.0024, 0.0020, 0.0031, 0.0025, 0.0025, 0.0022, 0.0024, 0.0030,
                        0.0027, 0.0027, 0.0027, 0.0026, 0.0023, 0.0024, 0.0023, 0.0024, 0.0023,
                        0.0020, 0.0022, 0.0031, 0.0021, 0.0028, 0.0024, 0.0022, 0.0020, 0.0025,
                        0.0020, 0.0021, 0.0020, 0.0026, 0.0037, 0.0020, 0.0023, 0.0025, 0.0022,
                        0.0023, 0.0024, 0.0023, 0.0024, 0.0027, 0.0024, 0.0024, 0.0025, 0.0024,
                        0.0026, 0.0022, 0.0019, 0.0022, 0.0019, 0.0021, 0.0027, 0.0022, 0.0025,
                        0.0022, 0.0019, 0.0025, 0.0026, 0.0024, 0.0022, 0.0023, 0.0022, 0.0022,
                        0.0019, 0.0030, 0.0021, 0.0026, 0.0026, 0.0018, 0.0022, 0.0025, 0.0023,
                        0.0020, 0.0025, 0.0036, 0.0024, 0.0020, 0.0025, 0.0023, 0.0028, 0.0032,
                        0.0024, 0.0025, 0.0028, 0.0028, 0.0026, 0.0026, 0.0030, 0.0022, 0.0024,
                        0.0020, 0.0028, 0.0024, 0.0022, 0.0026, 0.0022, 0.0020, 0.0028, 0.0024,
                        0.0020, 0.0023, 0.0024, 0.0022, 0.0025, 0.0020, 0.0022, 0.0021, 0.0032,
                        0.0023, 0.0025, 0.0025, 0.0026, 0.0021, 0.0021, 0.0026, 0.0030, 0.0024,
                        0.0021, 0.0020, 0.0023, 0.0029, 0.0023, 0.0021, 0.0024, 0.0025, 0.0025,
                        0.0023, 0.0027, 0.0028, 0.0022, 0.0024, 0.0033, 0.0024, 0.0025, 0.0020,
                        0.0020, 0.0021, 0.0025, 0.0021, 0.0025, 0.0027, 0.0023, 0.0020, 0.0019,
                        0.0019, 0.0025, 0.0020, 0.0026, 0.0020, 0.0024, 0.0022, 0.0024, 0.0026,
                        0.0020, 0.0024, 0.0019, 0.0019, 0.0022, 0.0025, 0.0024, 0.0021, 0.0021,
                        0.0026, 0.0023, 0.0024, 0.0020, 0.0026, 0.0026, 0.0023, 0.0024, 0.0028,
                        0.0026, 0.0020, 0.0028, 0.0023, 0.0024, 0.0027, 0.0028, 0.0018, 0.0022,
                        0.0017, 0.0022, 0.0025, 0.0020, 0.0016, 0.0026, 0.0025, 0.0022, 0.0021,
                        0.0021, 0.0023, 0.0022, 0.0022, 0.0026, 0.0029, 0.0020, 0.0019, 0.0026,
                        0.0021, 0.0026, 0.0025, 0.0025, 0.0021, 0.0031, 0.0023, 0.0021, 0.0027,
                        0.0027, 0.0022, 0.0025, 0.0019, 0.0024, 0.0025, 0.0022, 0.0020, 0.0029,
                        0.0026, 0.0020, 0.0025, 0.0022, 0.0019, 0.0019, 0.0019, 0.0025, 0.0025,
                        0.0021, 0.0027, 0.0020, 0.0026, 0.0022, 0.0021, 0.0023, 0.0026, 0.0022,
                        0.0023, 0.0021, 0.0025, 0.0027]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                  min_val=tensor([-0.2242, -0.2348, -0.2597, -0.2794, -0.1430, -0.2694, -0.4088, -0.2745,
                          -0.2631, -0.2813, -0.3008, -0.2618, -0.4029, -0.3152, -0.3252, -0.2403,
                          -0.3022, -0.2630, -0.3476, -0.3494, -0.3443, -0.3331, -0.2752, -0.3044,
                          -0.2904, -0.3025, -0.2720, -0.2591, -0.2844, -0.3968, -0.2717, -0.3621,
                          -0.2933, -0.2804, -0.2567, -0.3251, -0.2554, -0.2725, -0.2500, -0.3387,
                          -0.4734, -0.2518, -0.2941, -0.3190, -0.2838, -0.2517, -0.2944, -0.2926,
                          -0.3108, -0.3464, -0.2838, -0.2842, -0.3218, -0.3090, -0.2424, -0.2824,
                          -0.2147, -0.2793, -0.2434, -0.2322, -0.3506, -0.2829, -0.3217, -0.2743,
                          -0.2447, -0.3259, -0.3307, -0.2340, -0.2844, -0.2972, -0.2850, -0.2133,
                          -0.2340, -0.3903, -0.2220, -0.3354, -0.3327, -0.2309, -0.2647, -0.3136,
                          -0.2986, -0.2618, -0.3148, -0.4596, -0.3050, -0.2479, -0.3251, -0.2562,
                          -0.3540, -0.4159, -0.3011, -0.3146, -0.3545, -0.3571, -0.3332, -0.2539,
                          -0.3784, -0.2600, -0.3053, -0.2492, -0.3569, -0.3099, -0.2863, -0.3305,
                          -0.2871, -0.2539, -0.3108, -0.3040, -0.2398, -0.2878, -0.3060, -0.2764,
                          -0.3139, -0.2290, -0.2289, -0.2646, -0.4097, -0.2923, -0.2544, -0.3163,
                          -0.3296, -0.2745, -0.2625, -0.3271, -0.3869, -0.3064, -0.2626, -0.2508,
                          -0.2456, -0.3672, -0.2974, -0.1896, -0.2848, -0.3188, -0.2472, -0.2767,
                          -0.3445, -0.3575, -0.2829, -0.3120, -0.4166, -0.3030, -0.3245, -0.2583,
                          -0.2313, -0.2578, -0.3141, -0.2729, -0.2857, -0.3434, -0.2981, -0.2497,
                          -0.2430, -0.2492, -0.2352, -0.2598, -0.3304, -0.2571, -0.3099, -0.2769,
                          -0.3058, -0.3291, -0.2578, -0.3121, -0.2465, -0.2159, -0.2839, -0.3259,
                          -0.3106, -0.2008, -0.2642, -0.3312, -0.2906, -0.3103, -0.2518, -0.3352,
                          -0.3334, -0.2902, -0.3055, -0.3594, -0.3306, -0.2602, -0.3532, -0.2912,
                          -0.2892, -0.2079, -0.3641, -0.2335, -0.2532, -0.2238, -0.2775, -0.3260,
                          -0.2550, -0.2047, -0.3277, -0.3026, -0.2536, -0.2294, -0.2750, -0.2522,
                          -0.2814, -0.2873, -0.3323, -0.3768, -0.2512, -0.2431, -0.3271, -0.2736,
                          -0.3299, -0.3162, -0.3063, -0.2706, -0.3952, -0.2947, -0.2646, -0.3506,
                          -0.3449, -0.2838, -0.3176, -0.2286, -0.3103, -0.3154, -0.2845, -0.2521,
                          -0.3758, -0.3310, -0.2228, -0.2652, -0.2818, -0.2422, -0.2437, -0.2403,
                          -0.2643, -0.3226, -0.2308, -0.3414, -0.2213, -0.3352, -0.2857, -0.2708,
                          -0.2941, -0.3301, -0.2843, -0.2909, -0.2671, -0.3199, -0.3498]), max_val=tensor([0.2613, 0.1804, 0.2383, 0.2720, 0.3479, 0.2538, 0.2112, 0.2520, 0.3291,
                          0.2239, 0.2056, 0.2263, 0.1954, 0.2281, 0.2625, 0.2736, 0.2696, 0.3849,
                          0.1948, 0.1798, 0.2828, 0.2309, 0.2910, 0.2283, 0.2453, 0.2606, 0.2907,
                          0.2102, 0.2660, 0.2336, 0.2453, 0.1631, 0.3067, 0.2247, 0.2185, 0.2232,
                          0.1974, 0.2049, 0.2501, 0.2463, 0.1958, 0.1885, 0.2309, 0.1862, 0.2156,
                          0.2935, 0.3080, 0.2133, 0.2234, 0.1768, 0.2987, 0.3079, 0.2396, 0.2486,
                          0.3241, 0.2295, 0.2411, 0.2806, 0.1749, 0.2637, 0.2328, 0.2456, 0.1991,
                          0.2771, 0.1980, 0.3219, 0.2629, 0.2995, 0.2794, 0.2081, 0.2806, 0.2757,
                          0.2422, 0.2138, 0.2717, 0.2722, 0.1895, 0.2170, 0.2823, 0.1707, 0.2454,
                          0.2106, 0.2511, 0.2978, 0.2970, 0.2523, 0.2440, 0.2951, 0.2755, 0.2777,
                          0.2346, 0.2323, 0.2463, 0.2839, 0.2175, 0.3256, 0.2157, 0.2816, 0.2176,
                          0.2501, 0.2304, 0.2596, 0.2815, 0.2886, 0.1827, 0.2118, 0.3600, 0.1874,
                          0.2556, 0.2907, 0.2412, 0.2277, 0.2372, 0.2568, 0.2738, 0.2085, 0.2406,
                          0.2086, 0.3123, 0.2309, 0.2893, 0.2460, 0.2395, 0.1556, 0.2479, 0.2273,
                          0.2137, 0.2225, 0.2939, 0.2100, 0.2370, 0.2659, 0.3000, 0.2651, 0.3190,
                          0.2859, 0.3442, 0.2255, 0.2152, 0.1906, 0.3148, 0.2150, 0.2589, 0.1835,
                          0.2481, 0.2615, 0.2444, 0.2344, 0.3125, 0.3015, 0.1609, 0.2026, 0.2064,
                          0.2463, 0.3160, 0.2072, 0.2546, 0.2309, 0.2439, 0.2410, 0.2170, 0.2137,
                          0.2269, 0.2491, 0.2346, 0.2426, 0.2630, 0.2794, 0.2630, 0.2718, 0.2143,
                          0.3276, 0.2422, 0.3024, 0.2294, 0.2262, 0.1950, 0.2794, 0.2547, 0.3056,
                          0.2935, 0.2119, 0.2859, 0.2170, 0.3096, 0.3384, 0.2453, 0.1985, 0.2822,
                          0.1856, 0.2348, 0.2703, 0.2143, 0.1755, 0.2056, 0.3174, 0.2799, 0.2687,
                          0.1928, 0.2960, 0.2597, 0.2836, 0.1709, 0.3410, 0.2014, 0.2047, 0.2133,
                          0.2347, 0.2948, 0.2425, 0.3171, 0.2392, 0.3523, 0.2282, 0.2446, 0.3380,
                          0.2853, 0.2599, 0.2174, 0.2457, 0.2322, 0.2020, 0.2067, 0.2030, 0.2499,
                          0.2524, 0.2484, 0.3209, 0.2425, 0.2152, 0.1990, 0.2316, 0.3203, 0.2063,
                          0.2702, 0.2492, 0.2494, 0.2267, 0.2826, 0.2295, 0.2186, 0.1951, 0.2059,
                          0.2633, 0.2712, 0.3183, 0.1938])
                )
              )
              (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1159]), zero_point=tensor([76], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=-8.861878395080566, max_val=5.854861736297607)
              )
            )
            (1): BatchNorm2d(
              247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1033]), zero_point=tensor([65], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=-6.733070373535156, max_val=6.390249252319336)
              )
            )
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(
            247, 253, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0024, 0.0022, 0.0017, 0.0022, 0.0020, 0.0019, 0.0021, 0.0023, 0.0019,
                      0.0021, 0.0021, 0.0019, 0.0025, 0.0026, 0.0021, 0.0018, 0.0022, 0.0018,
                      0.0020, 0.0028, 0.0026, 0.0025, 0.0018, 0.0025, 0.0014, 0.0022, 0.0020,
                      0.0024, 0.0021, 0.0024, 0.0020, 0.0020, 0.0018, 0.0018, 0.0026, 0.0021,
                      0.0023, 0.0023, 0.0016, 0.0020, 0.0020, 0.0023, 0.0024, 0.0017, 0.0023,
                      0.0021, 0.0022, 0.0023, 0.0025, 0.0023, 0.0018, 0.0019, 0.0021, 0.0023,
                      0.0020, 0.0019, 0.0020, 0.0020, 0.0024, 0.0022, 0.0024, 0.0019, 0.0020,
                      0.0021, 0.0018, 0.0025, 0.0022, 0.0020, 0.0022, 0.0019, 0.0019, 0.0020,
                      0.0024, 0.0020, 0.0018, 0.0020, 0.0019, 0.0024, 0.0021, 0.0019, 0.0020,
                      0.0021, 0.0020, 0.0017, 0.0024, 0.0020, 0.0016, 0.0023, 0.0023, 0.0019,
                      0.0022, 0.0020, 0.0021, 0.0019, 0.0021, 0.0022, 0.0019, 0.0019, 0.0021,
                      0.0021, 0.0027, 0.0021, 0.0020, 0.0023, 0.0021, 0.0021, 0.0021, 0.0023,
                      0.0018, 0.0017, 0.0021, 0.0018, 0.0017, 0.0018, 0.0025, 0.0024, 0.0021,
                      0.0022, 0.0024, 0.0020, 0.0021, 0.0025, 0.0025, 0.0019, 0.0021, 0.0019,
                      0.0015, 0.0024, 0.0022, 0.0023, 0.0019, 0.0021, 0.0022, 0.0017, 0.0020,
                      0.0023, 0.0019, 0.0025, 0.0020, 0.0020, 0.0022, 0.0021, 0.0023, 0.0019,
                      0.0016, 0.0023, 0.0021, 0.0019, 0.0023, 0.0023, 0.0020, 0.0019, 0.0023,
                      0.0019, 0.0022, 0.0022, 0.0016, 0.0026, 0.0024, 0.0021, 0.0018, 0.0020,
                      0.0021, 0.0023, 0.0020, 0.0023, 0.0022, 0.0022, 0.0020, 0.0024, 0.0025,
                      0.0023, 0.0023, 0.0018, 0.0018, 0.0027, 0.0031, 0.0020, 0.0022, 0.0019,
                      0.0025, 0.0020, 0.0024, 0.0021, 0.0017, 0.0019, 0.0022, 0.0020, 0.0020,
                      0.0018, 0.0021, 0.0019, 0.0021, 0.0021, 0.0022, 0.0020, 0.0022, 0.0020,
                      0.0023, 0.0024, 0.0023, 0.0016, 0.0024, 0.0018, 0.0021, 0.0027, 0.0024,
                      0.0022, 0.0017, 0.0023, 0.0018, 0.0023, 0.0022, 0.0020, 0.0027, 0.0021,
                      0.0019, 0.0023, 0.0020, 0.0025, 0.0025, 0.0023, 0.0022, 0.0022, 0.0020,
                      0.0020, 0.0017, 0.0019, 0.0022, 0.0021, 0.0022, 0.0025, 0.0020, 0.0024,
                      0.0020, 0.0023, 0.0018, 0.0019, 0.0023, 0.0021, 0.0022, 0.0025, 0.0024,
                      0.0021, 0.0022, 0.0022, 0.0023, 0.0022, 0.0022, 0.0022, 0.0018, 0.0023,
                      0.0018]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2995, -0.2800, -0.2177, -0.2697, -0.2496, -0.2407, -0.2680, -0.2903,
                        -0.2481, -0.2647, -0.2598, -0.2369, -0.3232, -0.3347, -0.2752, -0.2325,
                        -0.2871, -0.2286, -0.2524, -0.3531, -0.2958, -0.2717, -0.2182, -0.3172,
                        -0.1793, -0.2829, -0.2547, -0.3100, -0.2710, -0.3021, -0.2622, -0.2580,
                        -0.2352, -0.2285, -0.3287, -0.2675, -0.2935, -0.2991, -0.2094, -0.2428,
                        -0.2522, -0.2884, -0.3095, -0.2131, -0.2971, -0.2666, -0.2841, -0.2969,
                        -0.3041, -0.2943, -0.2183, -0.2427, -0.2629, -0.2906, -0.2549, -0.2434,
                        -0.2595, -0.2549, -0.3105, -0.2815, -0.2514, -0.2436, -0.2605, -0.2640,
                        -0.2327, -0.3170, -0.2299, -0.2547, -0.2844, -0.2420, -0.2442, -0.2602,
                        -0.2553, -0.2511, -0.2321, -0.2574, -0.2296, -0.3010, -0.2675, -0.2401,
                        -0.2521, -0.2724, -0.2496, -0.2113, -0.3077, -0.2619, -0.1998, -0.3006,
                        -0.2890, -0.2443, -0.2861, -0.2531, -0.2682, -0.2396, -0.2735, -0.2844,
                        -0.2496, -0.2421, -0.2726, -0.2725, -0.3512, -0.2703, -0.2375, -0.2902,
                        -0.2610, -0.2555, -0.2508, -0.2985, -0.2096, -0.2127, -0.2646, -0.2219,
                        -0.2210, -0.2190, -0.2980, -0.3086, -0.2634, -0.2855, -0.2447, -0.2550,
                        -0.2498, -0.3247, -0.2537, -0.2427, -0.2631, -0.2429, -0.1969, -0.3065,
                        -0.2271, -0.2925, -0.2431, -0.2668, -0.2764, -0.2167, -0.2579, -0.2461,
                        -0.2194, -0.3147, -0.2520, -0.2599, -0.2756, -0.2624, -0.2921, -0.2374,
                        -0.2091, -0.2947, -0.2636, -0.2389, -0.2975, -0.2898, -0.2610, -0.2433,
                        -0.2896, -0.2454, -0.2857, -0.2583, -0.2104, -0.2363, -0.3021, -0.2698,
                        -0.2294, -0.2613, -0.2393, -0.2921, -0.2527, -0.2469, -0.2810, -0.2832,
                        -0.2565, -0.3098, -0.3175, -0.2929, -0.2886, -0.2192, -0.2349, -0.3448,
                        -0.3265, -0.2609, -0.2856, -0.2419, -0.3138, -0.2456, -0.2467, -0.2728,
                        -0.2147, -0.2420, -0.2821, -0.2621, -0.2603, -0.2051, -0.2743, -0.2430,
                        -0.2426, -0.2725, -0.2794, -0.2532, -0.2687, -0.2550, -0.2857, -0.3121,
                        -0.2888, -0.2043, -0.2820, -0.2253, -0.2698, -0.3396, -0.3100, -0.2764,
                        -0.2192, -0.2633, -0.2221, -0.2946, -0.2758, -0.2547, -0.3409, -0.2670,
                        -0.2463, -0.2924, -0.2621, -0.2851, -0.3172, -0.2894, -0.2818, -0.2791,
                        -0.1976, -0.2616, -0.2134, -0.2439, -0.2811, -0.2684, -0.2861, -0.2363,
                        -0.2291, -0.3088, -0.2561, -0.2935, -0.2264, -0.2424, -0.2950, -0.2656,
                        -0.2857, -0.3140, -0.3102, -0.2742, -0.2781, -0.2853, -0.2883, -0.2774,
                        -0.2846, -0.2780, -0.2269, -0.3005, -0.2357]), max_val=tensor([0.3037, 0.1953, 0.1792, 0.2747, 0.2405, 0.2029, 0.2235, 0.2663, 0.1930,
                        0.2542, 0.2607, 0.2198, 0.2210, 0.2500, 0.2139, 0.2160, 0.2557, 0.1871,
                        0.2252, 0.2600, 0.3326, 0.3175, 0.2262, 0.2288, 0.1537, 0.2508, 0.2546,
                        0.2753, 0.2458, 0.2409, 0.2564, 0.2319, 0.1782, 0.1987, 0.2549, 0.2208,
                        0.2704, 0.2672, 0.1807, 0.2543, 0.1825, 0.2677, 0.1810, 0.1883, 0.2224,
                        0.2097, 0.2141, 0.2225, 0.3181, 0.2187, 0.2334, 0.2186, 0.2047, 0.2259,
                        0.2230, 0.2461, 0.2299, 0.1782, 0.2497, 0.2576, 0.3065, 0.2148, 0.1984,
                        0.2078, 0.1894, 0.2208, 0.2762, 0.1793, 0.2497, 0.2343, 0.2314, 0.1979,
                        0.3006, 0.1896, 0.2298, 0.2343, 0.2415, 0.3036, 0.2339, 0.2313, 0.2032,
                        0.2371, 0.2448, 0.2021, 0.2464, 0.2040, 0.1780, 0.2153, 0.2642, 0.2183,
                        0.2103, 0.2266, 0.2025, 0.1885, 0.2539, 0.2269, 0.2125, 0.2233, 0.2621,
                        0.2345, 0.2585, 0.2443, 0.2486, 0.2345, 0.2637, 0.2675, 0.2642, 0.2441,
                        0.2343, 0.2215, 0.2165, 0.2279, 0.2050, 0.2329, 0.3238, 0.2113, 0.2251,
                        0.2120, 0.3035, 0.2391, 0.2717, 0.2191, 0.3183, 0.2296, 0.2280, 0.1686,
                        0.1892, 0.2610, 0.2837, 0.2428, 0.2365, 0.2384, 0.2609, 0.2076, 0.1650,
                        0.2889, 0.2358, 0.2694, 0.2227, 0.2582, 0.2687, 0.2543, 0.2272, 0.2366,
                        0.1879, 0.2839, 0.2429, 0.2354, 0.2376, 0.2561, 0.2097, 0.2214, 0.2909,
                        0.2350, 0.2482, 0.2742, 0.2043, 0.3273, 0.1985, 0.2607, 0.2285, 0.1905,
                        0.2712, 0.2202, 0.2252, 0.2977, 0.2151, 0.2397, 0.2229, 0.2863, 0.2348,
                        0.2285, 0.2294, 0.2306, 0.2171, 0.2891, 0.3907, 0.2486, 0.2248, 0.1673,
                        0.2851, 0.2479, 0.3053, 0.2525, 0.2125, 0.2314, 0.2600, 0.1945, 0.2463,
                        0.2225, 0.2709, 0.2278, 0.2612, 0.2609, 0.2377, 0.2571, 0.2829, 0.2006,
                        0.2866, 0.3014, 0.1945, 0.1851, 0.3026, 0.2197, 0.2005, 0.2307, 0.2010,
                        0.2234, 0.1965, 0.2932, 0.2257, 0.2538, 0.2640, 0.2372, 0.2641, 0.2387,
                        0.2256, 0.2923, 0.2126, 0.3152, 0.2149, 0.2707, 0.2212, 0.2460, 0.2581,
                        0.2487, 0.1668, 0.2471, 0.2788, 0.2590, 0.2559, 0.3213, 0.2532, 0.2534,
                        0.1748, 0.2389, 0.2064, 0.2421, 0.2266, 0.2472, 0.2246, 0.2619, 0.2554,
                        0.2387, 0.2161, 0.2216, 0.2243, 0.2267, 0.2097, 0.2558, 0.2300, 0.2356,
                        0.2335])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.2655]), zero_point=tensor([80], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-21.321176528930664, max_val=12.392306327819824)
            )
          )
          (bn1): BatchNorm2d(
            253, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0833]), zero_point=tensor([63], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-5.281262397766113, max_val=5.30015230178833)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            253, 247, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0017, 0.0018, 0.0011, 0.0019, 0.0019, 0.0016, 0.0024, 0.0015, 0.0018,
                      0.0018, 0.0020, 0.0023, 0.0016, 0.0023, 0.0018, 0.0025, 0.0021, 0.0020,
                      0.0015, 0.0024, 0.0013, 0.0021, 0.0018, 0.0017, 0.0018, 0.0021, 0.0021,
                      0.0018, 0.0019, 0.0017, 0.0027, 0.0026, 0.0019, 0.0026, 0.0017, 0.0017,
                      0.0019, 0.0018, 0.0018, 0.0015, 0.0018, 0.0018, 0.0018, 0.0022, 0.0023,
                      0.0025, 0.0021, 0.0015, 0.0019, 0.0012, 0.0013, 0.0016, 0.0017, 0.0015,
                      0.0020, 0.0024, 0.0015, 0.0023, 0.0018, 0.0020, 0.0025, 0.0016, 0.0025,
                      0.0020, 0.0024, 0.0020, 0.0025, 0.0018, 0.0029, 0.0020, 0.0019, 0.0018,
                      0.0025, 0.0021, 0.0023, 0.0024, 0.0016, 0.0018, 0.0021, 0.0022, 0.0025,
                      0.0021, 0.0020, 0.0016, 0.0024, 0.0019, 0.0022, 0.0021, 0.0017, 0.0022,
                      0.0014, 0.0019, 0.0016, 0.0028, 0.0026, 0.0022, 0.0017, 0.0021, 0.0020,
                      0.0021, 0.0015, 0.0015, 0.0019, 0.0021, 0.0019, 0.0019, 0.0021, 0.0023,
                      0.0023, 0.0023, 0.0026, 0.0016, 0.0020, 0.0018, 0.0021, 0.0015, 0.0021,
                      0.0018, 0.0023, 0.0017, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0015,
                      0.0023, 0.0022, 0.0020, 0.0021, 0.0014, 0.0020, 0.0018, 0.0017, 0.0021,
                      0.0016, 0.0020, 0.0024, 0.0022, 0.0022, 0.0016, 0.0019, 0.0019, 0.0025,
                      0.0020, 0.0035, 0.0017, 0.0023, 0.0019, 0.0025, 0.0021, 0.0018, 0.0017,
                      0.0020, 0.0019, 0.0016, 0.0019, 0.0020, 0.0023, 0.0023, 0.0014, 0.0020,
                      0.0017, 0.0016, 0.0019, 0.0022, 0.0019, 0.0018, 0.0019, 0.0015, 0.0020,
                      0.0015, 0.0016, 0.0017, 0.0013, 0.0016, 0.0020, 0.0015, 0.0017, 0.0016,
                      0.0020, 0.0021, 0.0019, 0.0016, 0.0026, 0.0020, 0.0018, 0.0027, 0.0017,
                      0.0016, 0.0018, 0.0015, 0.0018, 0.0019, 0.0018, 0.0020, 0.0025, 0.0018,
                      0.0020, 0.0021, 0.0022, 0.0020, 0.0020, 0.0013, 0.0024, 0.0017, 0.0020,
                      0.0018, 0.0021, 0.0023, 0.0022, 0.0019, 0.0020, 0.0020, 0.0019, 0.0019,
                      0.0019, 0.0019, 0.0024, 0.0021, 0.0021, 0.0020, 0.0016, 0.0018, 0.0018,
                      0.0015, 0.0020, 0.0025, 0.0019, 0.0023, 0.0023, 0.0018, 0.0019, 0.0022,
                      0.0023, 0.0015, 0.0017, 0.0020, 0.0020, 0.0018, 0.0018, 0.0020, 0.0016,
                      0.0025, 0.0023, 0.0020, 0.0020]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2178, -0.2203, -0.1195, -0.2260, -0.2253, -0.2027, -0.3025, -0.1923,
                        -0.2221, -0.2197, -0.2458, -0.2153, -0.2031, -0.2471, -0.1369, -0.3153,
                        -0.2514, -0.2520, -0.1927, -0.3075, -0.1550, -0.2750, -0.1794, -0.2084,
                        -0.2269, -0.2563, -0.2501, -0.2306, -0.2477, -0.2225, -0.3502, -0.3275,
                        -0.2393, -0.2283, -0.2162, -0.2169, -0.2464, -0.2056, -0.1741, -0.1977,
                        -0.2334, -0.2306, -0.2273, -0.2835, -0.1756, -0.3186, -0.2630, -0.1814,
                        -0.1628, -0.1562, -0.1546, -0.1948, -0.2037, -0.1353, -0.2287, -0.2170,
                        -0.1904, -0.3007, -0.2124, -0.2134, -0.3250, -0.1835, -0.2162, -0.2512,
                        -0.3064, -0.1854, -0.2671, -0.1805, -0.2531, -0.2501, -0.2446, -0.2348,
                        -0.3153, -0.2520, -0.2137, -0.2253, -0.2105, -0.2047, -0.1855, -0.2861,
                        -0.3165, -0.2692, -0.2027, -0.1868, -0.3059, -0.2388, -0.2837, -0.2585,
                        -0.2134, -0.2628, -0.1822, -0.2424, -0.1804, -0.2702, -0.2649, -0.2770,
                        -0.2120, -0.2625, -0.2393, -0.1825, -0.1906, -0.1864, -0.2319, -0.2526,
                        -0.2444, -0.1816, -0.2704, -0.2528, -0.2437, -0.2897, -0.2481, -0.2000,
                        -0.2279, -0.1955, -0.2721, -0.1442, -0.2683, -0.2256, -0.2920, -0.2117,
                        -0.2490, -0.2432, -0.2488, -0.2475, -0.2414, -0.1879, -0.2801, -0.2840,
                        -0.2600, -0.2048, -0.1708, -0.1540, -0.2273, -0.1881, -0.2208, -0.1996,
                        -0.2553, -0.3034, -0.2851, -0.2782, -0.2057, -0.2445, -0.2231, -0.2233,
                        -0.2143, -0.2427, -0.2219, -0.2974, -0.2368, -0.3232, -0.2724, -0.2325,
                        -0.2075, -0.2602, -0.1996, -0.1993, -0.1700, -0.2597, -0.2462, -0.2831,
                        -0.1794, -0.2489, -0.2223, -0.1429, -0.1937, -0.2829, -0.2291, -0.2131,
                        -0.1829, -0.1889, -0.2386, -0.1943, -0.2058, -0.1846, -0.1707, -0.1721,
                        -0.2588, -0.1940, -0.2195, -0.1747, -0.2617, -0.1833, -0.2475, -0.2094,
                        -0.2315, -0.1826, -0.2294, -0.2304, -0.1819, -0.1563, -0.2321, -0.1983,
                        -0.2276, -0.2369, -0.2112, -0.2588, -0.1697, -0.1926, -0.2501, -0.2642,
                        -0.2330, -0.2512, -0.2133, -0.1669, -0.2148, -0.1942, -0.1647, -0.2292,
                        -0.2357, -0.2930, -0.1867, -0.2382, -0.2562, -0.2517, -0.2445, -0.2392,
                        -0.2457, -0.2401, -0.2583, -0.2464, -0.1705, -0.2598, -0.1436, -0.2139,
                        -0.2336, -0.1888, -0.2511, -0.2467, -0.2412, -0.2052, -0.2984, -0.2313,
                        -0.2299, -0.2503, -0.2878, -0.1974, -0.2146, -0.2606, -0.2349, -0.2141,
                        -0.2323, -0.2504, -0.1988, -0.3189, -0.2903, -0.2612, -0.2547]), max_val=tensor([0.1878, 0.2239, 0.1340, 0.2415, 0.2378, 0.1777, 0.3003, 0.1787, 0.2287,
                        0.2266, 0.2487, 0.2919, 0.2072, 0.2884, 0.2334, 0.2570, 0.2673, 0.1364,
                        0.1954, 0.2583, 0.1613, 0.2140, 0.2319, 0.2208, 0.2156, 0.2611, 0.2610,
                        0.1946, 0.2202, 0.1786, 0.2677, 0.2881, 0.2315, 0.3246, 0.2013, 0.1377,
                        0.1689, 0.2280, 0.2329, 0.1951, 0.2031, 0.1564, 0.2241, 0.2337, 0.2961,
                        0.3222, 0.2040, 0.1856, 0.2421, 0.1245, 0.1605, 0.1973, 0.2208, 0.1854,
                        0.2545, 0.3048, 0.1693, 0.2610, 0.2254, 0.2526, 0.2452, 0.2059, 0.3116,
                        0.2507, 0.2613, 0.2590, 0.3188, 0.2246, 0.3706, 0.1932, 0.1999, 0.2068,
                        0.2545, 0.2674, 0.2878, 0.3010, 0.1800, 0.2303, 0.2679, 0.2439, 0.2803,
                        0.2257, 0.2594, 0.2015, 0.2379, 0.1646, 0.2678, 0.2629, 0.1656, 0.2766,
                        0.1760, 0.1967, 0.2082, 0.3501, 0.3341, 0.2200, 0.2091, 0.2105, 0.2522,
                        0.2690, 0.1722, 0.1642, 0.2396, 0.2702, 0.1536, 0.2412, 0.2132, 0.2878,
                        0.2884, 0.2716, 0.3248, 0.1678, 0.2483, 0.2309, 0.2129, 0.1960, 0.1771,
                        0.1798, 0.2588, 0.1996, 0.2139, 0.1715, 0.1832, 0.1766, 0.2361, 0.1699,
                        0.2962, 0.2723, 0.2222, 0.2672, 0.1747, 0.2603, 0.2198, 0.2141, 0.2607,
                        0.2086, 0.2165, 0.3021, 0.2184, 0.2295, 0.2065, 0.1835, 0.2356, 0.3209,
                        0.2513, 0.4491, 0.2003, 0.2147, 0.2228, 0.2919, 0.1953, 0.2218, 0.2156,
                        0.2159, 0.2469, 0.2073, 0.2390, 0.2302, 0.2934, 0.2955, 0.1384, 0.2487,
                        0.2084, 0.1972, 0.2412, 0.2738, 0.2375, 0.2291, 0.2475, 0.1475, 0.2563,
                        0.1791, 0.1966, 0.2149, 0.1714, 0.2011, 0.2552, 0.1799, 0.1270, 0.2070,
                        0.2273, 0.2679, 0.2364, 0.2077, 0.3335, 0.2504, 0.2125, 0.3414, 0.2135,
                        0.2077, 0.2176, 0.1821, 0.2275, 0.1551, 0.2246, 0.2260, 0.3217, 0.2309,
                        0.2184, 0.2168, 0.2805, 0.2577, 0.2477, 0.1616, 0.3047, 0.2173, 0.2492,
                        0.2192, 0.2625, 0.2075, 0.2818, 0.2217, 0.2287, 0.2516, 0.2243, 0.1643,
                        0.2323, 0.1869, 0.3010, 0.2651, 0.2665, 0.2437, 0.1986, 0.2226, 0.2028,
                        0.1927, 0.2217, 0.3169, 0.1809, 0.2859, 0.1967, 0.1994, 0.2438, 0.2770,
                        0.2874, 0.1873, 0.2096, 0.2068, 0.2497, 0.2301, 0.2224, 0.2191, 0.2056,
                        0.1796, 0.2700, 0.1719, 0.2327])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.6079]), zero_point=tensor([74], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-44.68861770629883, max_val=32.51748275756836)
            )
          )
          (bn2): BatchNorm2d(
            247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0104]), zero_point=tensor([61], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.6285780668258667, max_val=0.686379611492157)
            )
          )
          (act2): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(
            247, 509, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0022, 0.0023, 0.0021, 0.0019, 0.0021, 0.0020, 0.0028, 0.0021, 0.0024,
                      0.0024, 0.0021, 0.0026, 0.0018, 0.0021, 0.0025, 0.0025, 0.0023, 0.0025,
                      0.0024, 0.0023, 0.0021, 0.0024, 0.0027, 0.0022, 0.0024, 0.0024, 0.0021,
                      0.0023, 0.0022, 0.0025, 0.0022, 0.0021, 0.0024, 0.0025, 0.0020, 0.0027,
                      0.0024, 0.0025, 0.0021, 0.0025, 0.0024, 0.0020, 0.0024, 0.0018, 0.0023,
                      0.0029, 0.0025, 0.0028, 0.0022, 0.0023, 0.0021, 0.0022, 0.0020, 0.0023,
                      0.0031, 0.0024, 0.0021, 0.0023, 0.0023, 0.0025, 0.0024, 0.0022, 0.0021,
                      0.0024, 0.0021, 0.0027, 0.0024, 0.0022, 0.0031, 0.0025, 0.0022, 0.0020,
                      0.0023, 0.0029, 0.0026, 0.0020, 0.0026, 0.0026, 0.0023, 0.0020, 0.0019,
                      0.0021, 0.0020, 0.0021, 0.0021, 0.0018, 0.0023, 0.0020, 0.0029, 0.0023,
                      0.0021, 0.0024, 0.0022, 0.0025, 0.0027, 0.0021, 0.0028, 0.0022, 0.0021,
                      0.0024, 0.0030, 0.0020, 0.0028, 0.0027, 0.0019, 0.0021, 0.0029, 0.0019,
                      0.0022, 0.0025, 0.0024, 0.0032, 0.0027, 0.0027, 0.0023, 0.0025, 0.0028,
                      0.0019, 0.0023, 0.0021, 0.0025, 0.0025, 0.0022, 0.0024, 0.0022, 0.0029,
                      0.0022, 0.0030, 0.0021, 0.0025, 0.0027, 0.0027, 0.0019, 0.0026, 0.0022,
                      0.0020, 0.0022, 0.0026, 0.0024, 0.0021, 0.0023, 0.0028, 0.0023, 0.0029,
                      0.0022, 0.0021, 0.0023, 0.0024, 0.0026, 0.0019, 0.0022, 0.0020, 0.0021,
                      0.0022, 0.0024, 0.0027, 0.0019, 0.0021, 0.0023, 0.0033, 0.0026, 0.0022,
                      0.0024, 0.0016, 0.0024, 0.0023, 0.0024, 0.0025, 0.0022, 0.0025, 0.0021,
                      0.0024, 0.0014, 0.0021, 0.0023, 0.0021, 0.0033, 0.0018, 0.0022, 0.0023,
                      0.0023, 0.0025, 0.0023, 0.0021, 0.0021, 0.0020, 0.0025, 0.0022, 0.0022,
                      0.0021, 0.0019, 0.0022, 0.0027, 0.0027, 0.0021, 0.0021, 0.0023, 0.0026,
                      0.0022, 0.0023, 0.0019, 0.0023, 0.0020, 0.0024, 0.0023, 0.0025, 0.0023,
                      0.0021, 0.0024, 0.0021, 0.0022, 0.0025, 0.0022, 0.0023, 0.0019, 0.0034,
                      0.0023, 0.0022, 0.0024, 0.0022, 0.0024, 0.0027, 0.0024, 0.0021, 0.0024,
                      0.0017, 0.0021, 0.0024, 0.0018, 0.0024, 0.0018, 0.0024, 0.0021, 0.0023,
                      0.0021, 0.0024, 0.0021, 0.0020, 0.0022, 0.0023, 0.0018, 0.0023, 0.0023,
                      0.0025, 0.0022, 0.0020, 0.0022, 0.0019, 0.0021, 0.0026, 0.0022, 0.0017,
                      0.0021, 0.0019, 0.0026, 0.0022, 0.0021, 0.0027, 0.0021, 0.0019, 0.0022,
                      0.0021, 0.0024, 0.0025, 0.0027, 0.0022, 0.0023, 0.0025, 0.0022, 0.0025,
                      0.0026, 0.0025, 0.0026, 0.0021, 0.0019, 0.0023, 0.0026, 0.0024, 0.0027,
                      0.0023, 0.0024, 0.0022, 0.0024, 0.0022, 0.0024, 0.0029, 0.0026, 0.0027,
                      0.0022, 0.0025, 0.0020, 0.0023, 0.0022, 0.0021, 0.0023, 0.0024, 0.0024,
                      0.0022, 0.0017, 0.0027, 0.0020, 0.0027, 0.0024, 0.0020, 0.0021, 0.0021,
                      0.0018, 0.0023, 0.0023, 0.0018, 0.0020, 0.0025, 0.0022, 0.0018, 0.0026,
                      0.0021, 0.0017, 0.0021, 0.0031, 0.0024, 0.0024, 0.0023, 0.0019, 0.0021,
                      0.0022, 0.0024, 0.0019, 0.0028, 0.0021, 0.0021, 0.0025, 0.0029, 0.0020,
                      0.0027, 0.0026, 0.0023, 0.0020, 0.0019, 0.0018, 0.0026, 0.0024, 0.0026,
                      0.0025, 0.0022, 0.0023, 0.0024, 0.0025, 0.0024, 0.0022, 0.0023, 0.0025,
                      0.0024, 0.0014, 0.0021, 0.0022, 0.0026, 0.0024, 0.0025, 0.0019, 0.0021,
                      0.0026, 0.0020, 0.0022, 0.0026, 0.0020, 0.0025, 0.0026, 0.0021, 0.0024,
                      0.0024, 0.0020, 0.0017, 0.0021, 0.0022, 0.0020, 0.0024, 0.0020, 0.0026,
                      0.0020, 0.0020, 0.0024, 0.0025, 0.0024, 0.0021, 0.0020, 0.0023, 0.0026,
                      0.0020, 0.0020, 0.0018, 0.0022, 0.0019, 0.0019, 0.0025, 0.0024, 0.0023,
                      0.0021, 0.0025, 0.0023, 0.0028, 0.0023, 0.0018, 0.0020, 0.0028, 0.0026,
                      0.0030, 0.0023, 0.0020, 0.0027, 0.0027, 0.0019, 0.0019, 0.0023, 0.0025,
                      0.0023, 0.0021, 0.0021, 0.0018, 0.0021, 0.0022, 0.0025, 0.0024, 0.0025,
                      0.0025, 0.0018, 0.0023, 0.0019, 0.0020, 0.0026, 0.0022, 0.0016, 0.0022,
                      0.0019, 0.0021, 0.0024, 0.0021, 0.0023, 0.0023, 0.0021, 0.0020, 0.0022,
                      0.0018, 0.0021, 0.0029, 0.0024, 0.0022, 0.0022, 0.0025, 0.0020, 0.0027,
                      0.0028, 0.0030, 0.0026, 0.0021, 0.0025, 0.0022, 0.0021, 0.0024, 0.0021,
                      0.0024, 0.0022, 0.0018, 0.0030, 0.0027, 0.0023, 0.0022, 0.0023, 0.0021,
                      0.0020, 0.0024, 0.0024, 0.0025, 0.0019, 0.0026, 0.0021, 0.0021, 0.0022,
                      0.0025, 0.0025, 0.0024, 0.0022, 0.0026, 0.0025, 0.0020, 0.0022, 0.0025,
                      0.0022, 0.0023, 0.0020, 0.0025, 0.0023, 0.0022, 0.0025, 0.0022, 0.0030,
                      0.0026, 0.0026, 0.0026, 0.0024, 0.0024, 0.0021, 0.0023, 0.0019, 0.0020,
                      0.0026, 0.0021, 0.0017, 0.0025, 0.0027]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2763, -0.2957, -0.2704, -0.2359, -0.2013, -0.2582, -0.3648, -0.2614,
                        -0.3120, -0.3096, -0.2633, -0.3288, -0.2287, -0.2704, -0.3260, -0.3195,
                        -0.2539, -0.3197, -0.3110, -0.2960, -0.2670, -0.2586, -0.3429, -0.2550,
                        -0.3013, -0.3052, -0.2682, -0.2904, -0.2843, -0.3257, -0.2772, -0.2417,
                        -0.3019, -0.3167, -0.2590, -0.2693, -0.3034, -0.3228, -0.2635, -0.3160,
                        -0.3050, -0.2411, -0.3090, -0.2221, -0.3001, -0.2810, -0.3147, -0.3565,
                        -0.2775, -0.3004, -0.2569, -0.2772, -0.2590, -0.2850, -0.4006, -0.3045,
                        -0.2748, -0.3005, -0.2986, -0.2978, -0.3097, -0.2363, -0.2746, -0.3113,
                        -0.2734, -0.3426, -0.3037, -0.2803, -0.4003, -0.2454, -0.2446, -0.2179,
                        -0.2972, -0.3741, -0.3320, -0.2465, -0.2985, -0.3364, -0.2925, -0.2600,
                        -0.2474, -0.2714, -0.2550, -0.2665, -0.2710, -0.2215, -0.2880, -0.2497,
                        -0.3736, -0.2409, -0.2716, -0.2837, -0.2765, -0.3188, -0.3447, -0.2715,
                        -0.3563, -0.2862, -0.2661, -0.2830, -0.3878, -0.2502, -0.3023, -0.3364,
                        -0.2413, -0.2721, -0.3715, -0.2410, -0.2329, -0.3173, -0.3110, -0.4041,
                        -0.2624, -0.2991, -0.2923, -0.2956, -0.3630, -0.2350, -0.2989, -0.2654,
                        -0.3171, -0.3175, -0.2808, -0.3132, -0.2831, -0.3663, -0.2877, -0.3780,
                        -0.2674, -0.3167, -0.3402, -0.3467, -0.2404, -0.3328, -0.2539, -0.2589,
                        -0.2630, -0.3370, -0.2653, -0.2667, -0.2899, -0.3629, -0.2862, -0.3702,
                        -0.2852, -0.2730, -0.2946, -0.3120, -0.3247, -0.2469, -0.2852, -0.2564,
                        -0.2657, -0.2447, -0.3026, -0.2486, -0.2423, -0.2702, -0.2889, -0.4174,
                        -0.3318, -0.2445, -0.3095, -0.2107, -0.3095, -0.2881, -0.3079, -0.2586,
                        -0.2780, -0.3152, -0.2743, -0.3015, -0.1806, -0.2648, -0.2883, -0.2251,
                        -0.4226, -0.2320, -0.2782, -0.2981, -0.2795, -0.3181, -0.2980, -0.2636,
                        -0.2737, -0.2543, -0.3157, -0.2823, -0.2828, -0.2635, -0.2388, -0.2853,
                        -0.3416, -0.3411, -0.2652, -0.2653, -0.2964, -0.3355, -0.2787, -0.2327,
                        -0.2494, -0.2811, -0.2519, -0.3113, -0.2520, -0.3228, -0.2586, -0.2705,
                        -0.3008, -0.2703, -0.2774, -0.3228, -0.2792, -0.2817, -0.2481, -0.4301,
                        -0.2969, -0.2861, -0.2664, -0.2299, -0.2610, -0.3410, -0.2419, -0.2624,
                        -0.3009, -0.2176, -0.2665, -0.3125, -0.2059, -0.3075, -0.2291, -0.2421,
                        -0.2710, -0.2743, -0.2681, -0.3075, -0.2670, -0.2455, -0.2805, -0.2936,
                        -0.2299, -0.2965, -0.2957, -0.3262, -0.2755, -0.2620, -0.2759, -0.2293,
                        -0.2708, -0.3387, -0.2733, -0.2222, -0.2473, -0.2476, -0.3373, -0.2772,
                        -0.2283, -0.3468, -0.2618, -0.2465, -0.2828, -0.2662, -0.3126, -0.3252,
                        -0.3499, -0.2847, -0.2927, -0.3239, -0.2817, -0.3189, -0.3325, -0.2925,
                        -0.3385, -0.2408, -0.2464, -0.2988, -0.3349, -0.3121, -0.3407, -0.2952,
                        -0.3132, -0.2313, -0.3026, -0.2853, -0.2485, -0.3652, -0.3302, -0.3429,
                        -0.2217, -0.3198, -0.2286, -0.2910, -0.2533, -0.2410, -0.2982, -0.3105,
                        -0.3114, -0.2869, -0.2167, -0.3409, -0.2588, -0.2490, -0.3076, -0.2575,
                        -0.2698, -0.2681, -0.2341, -0.2979, -0.2958, -0.2265, -0.2604, -0.2604,
                        -0.2392, -0.2334, -0.3294, -0.2615, -0.2210, -0.2642, -0.2881, -0.2831,
                        -0.3008, -0.3003, -0.2445, -0.2655, -0.2554, -0.3066, -0.2487, -0.3553,
                        -0.2647, -0.2708, -0.3193, -0.3661, -0.2269, -0.3396, -0.3278, -0.2967,
                        -0.2558, -0.2441, -0.2182, -0.3387, -0.3033, -0.3281, -0.3138, -0.2832,
                        -0.2538, -0.3028, -0.3171, -0.2661, -0.2821, -0.2957, -0.3185, -0.2755,
                        -0.1791, -0.2723, -0.2340, -0.3307, -0.2603, -0.3180, -0.2376, -0.2729,
                        -0.3286, -0.2618, -0.2387, -0.3379, -0.2511, -0.3219, -0.3328, -0.2169,
                        -0.3061, -0.3122, -0.2547, -0.2161, -0.2521, -0.2771, -0.2516, -0.3070,
                        -0.2579, -0.3366, -0.2514, -0.2576, -0.2414, -0.3197, -0.3050, -0.2589,
                        -0.2579, -0.3006, -0.3284, -0.2547, -0.2404, -0.2215, -0.2871, -0.2378,
                        -0.2486, -0.3116, -0.3116, -0.2902, -0.2625, -0.3258, -0.2461, -0.3531,
                        -0.2998, -0.2244, -0.2538, -0.3413, -0.3283, -0.3806, -0.2883, -0.2567,
                        -0.3393, -0.3164, -0.2358, -0.2399, -0.3006, -0.2627, -0.2984, -0.2650,
                        -0.2697, -0.2344, -0.2680, -0.2843, -0.3024, -0.3042, -0.2846, -0.2874,
                        -0.2292, -0.2720, -0.2197, -0.2540, -0.3056, -0.2808, -0.2085, -0.2606,
                        -0.2491, -0.2679, -0.3085, -0.2691, -0.2923, -0.2913, -0.2713, -0.2364,
                        -0.2798, -0.2326, -0.2628, -0.3718, -0.2927, -0.2768, -0.2642, -0.2726,
                        -0.2317, -0.3494, -0.2754, -0.3803, -0.3309, -0.2720, -0.3194, -0.2324,
                        -0.2677, -0.3093, -0.2711, -0.3128, -0.2865, -0.2364, -0.3804, -0.3432,
                        -0.2554, -0.2875, -0.2670, -0.2702, -0.2530, -0.3045, -0.2955, -0.3177,
                        -0.2434, -0.3369, -0.2706, -0.2649, -0.2816, -0.3261, -0.3208, -0.3016,
                        -0.2790, -0.3272, -0.3142, -0.2550, -0.2838, -0.3164, -0.2770, -0.2920,
                        -0.2550, -0.3243, -0.2928, -0.2777, -0.3257, -0.2764, -0.2484, -0.3299,
                        -0.3357, -0.2374, -0.2884, -0.3055, -0.2678, -0.2601, -0.2395, -0.2573,
                        -0.3012, -0.2724, -0.2235, -0.2800, -0.3401]), max_val=tensor([0.2579, 0.2656, 0.2433, 0.2456, 0.2708, 0.2206, 0.2783, 0.2647, 0.2681,
                        0.2863, 0.2577, 0.2476, 0.2277, 0.1816, 0.2867, 0.2576, 0.2903, 0.2458,
                        0.2970, 0.2820, 0.2614, 0.3035, 0.2062, 0.2834, 0.2184, 0.2777, 0.2227,
                        0.2608, 0.2693, 0.2250, 0.2221, 0.2703, 0.2017, 0.2499, 0.2227, 0.3419,
                        0.2456, 0.2499, 0.2011, 0.2605, 0.1877, 0.2482, 0.2816, 0.2228, 0.2481,
                        0.3720, 0.3155, 0.2941, 0.2391, 0.2592, 0.2725, 0.2434, 0.2581, 0.2889,
                        0.3313, 0.2944, 0.2507, 0.2885, 0.2097, 0.3161, 0.2049, 0.2817, 0.2602,
                        0.2716, 0.1870, 0.2309, 0.1897, 0.2854, 0.2205, 0.3162, 0.2731, 0.2494,
                        0.2460, 0.3204, 0.2294, 0.2601, 0.3250, 0.2248, 0.2421, 0.2235, 0.2202,
                        0.2428, 0.2353, 0.1708, 0.2522, 0.2317, 0.2294, 0.1837, 0.2409, 0.2942,
                        0.2699, 0.3094, 0.2144, 0.2135, 0.2729, 0.2304, 0.2592, 0.2458, 0.2359,
                        0.3031, 0.3095, 0.2452, 0.3559, 0.3455, 0.2157, 0.2110, 0.2162, 0.2241,
                        0.2839, 0.2735, 0.3106, 0.2821, 0.3377, 0.3369, 0.2918, 0.3179, 0.2376,
                        0.2406, 0.2391, 0.2166, 0.2933, 0.2781, 0.2439, 0.1926, 0.2622, 0.2055,
                        0.2271, 0.2782, 0.2452, 0.2840, 0.2642, 0.2830, 0.2451, 0.2551, 0.2777,
                        0.2168, 0.2784, 0.3362, 0.3075, 0.2411, 0.2785, 0.2323, 0.2918, 0.2278,
                        0.2030, 0.2525, 0.2661, 0.2649, 0.3308, 0.2074, 0.2596, 0.2406, 0.2090,
                        0.2743, 0.2813, 0.3420, 0.2194, 0.2548, 0.2457, 0.2906, 0.3115, 0.2778,
                        0.2522, 0.1883, 0.2211, 0.2816, 0.2601, 0.3161, 0.2824, 0.3043, 0.2252,
                        0.2408, 0.1739, 0.2444, 0.1849, 0.2660, 0.1911, 0.2299, 0.2318, 0.2012,
                        0.2983, 0.2668, 0.2249, 0.2309, 0.2689, 0.2465, 0.2282, 0.2280, 0.2109,
                        0.2022, 0.1874, 0.1880, 0.2153, 0.2304, 0.2726, 0.1462, 0.2673, 0.2436,
                        0.2755, 0.2859, 0.2234, 0.2898, 0.2309, 0.2405, 0.2967, 0.2998, 0.2894,
                        0.2141, 0.2222, 0.2264, 0.2350, 0.2419, 0.2258, 0.2956, 0.2036, 0.2431,
                        0.2699, 0.2453, 0.3080, 0.2803, 0.3060, 0.2777, 0.3049, 0.2373, 0.3000,
                        0.1925, 0.2421, 0.2948, 0.2248, 0.2626, 0.1851, 0.3085, 0.2358, 0.2970,
                        0.2384, 0.3093, 0.2332, 0.2502, 0.2319, 0.2461, 0.1860, 0.2433, 0.2242,
                        0.2309, 0.2337, 0.2583, 0.2593, 0.2458, 0.2642, 0.3119, 0.2732, 0.2113,
                        0.2630, 0.1908, 0.2754, 0.2777, 0.2717, 0.2700, 0.2611, 0.2247, 0.2023,
                        0.2315, 0.2238, 0.2519, 0.2738, 0.2068, 0.2543, 0.2499, 0.2669, 0.2514,
                        0.2646, 0.3181, 0.2352, 0.2628, 0.2273, 0.2882, 0.2952, 0.2364, 0.1910,
                        0.2502, 0.2607, 0.2732, 0.2736, 0.1853, 0.3080, 0.2632, 0.2568, 0.2406,
                        0.2839, 0.2479, 0.2495, 0.2304, 0.2789, 0.2680, 0.2188, 0.2658, 0.2704,
                        0.2430, 0.1716, 0.2094, 0.2258, 0.3402, 0.2394, 0.1751, 0.2312, 0.2132,
                        0.1820, 0.2981, 0.1971, 0.2201, 0.2387, 0.3126, 0.2761, 0.2238, 0.2717,
                        0.2708, 0.2120, 0.2311, 0.3999, 0.3001, 0.2568, 0.2367, 0.2344, 0.2463,
                        0.2764, 0.2068, 0.1740, 0.1932, 0.2650, 0.2601, 0.2343, 0.3704, 0.2542,
                        0.2471, 0.2531, 0.2205, 0.2452, 0.2057, 0.2230, 0.3020, 0.2674, 0.2643,
                        0.2768, 0.2009, 0.2935, 0.2230, 0.1777, 0.3017, 0.2272, 0.2948, 0.3203,
                        0.3084, 0.1613, 0.2692, 0.2770, 0.2520, 0.3028, 0.2449, 0.2297, 0.2225,
                        0.2514, 0.2425, 0.2841, 0.2531, 0.2470, 0.2317, 0.2665, 0.2648, 0.2433,
                        0.2668, 0.1895, 0.1964, 0.2624, 0.2104, 0.2088, 0.2689, 0.2339, 0.2795,
                        0.2039, 0.2323, 0.3018, 0.2341, 0.2384, 0.2711, 0.2526, 0.2518, 0.2095,
                        0.2073, 0.2487, 0.2238, 0.2443, 0.2209, 0.2190, 0.3187, 0.2573, 0.2089,
                        0.2647, 0.2643, 0.2937, 0.3066, 0.2484, 0.1952, 0.2319, 0.3500, 0.2791,
                        0.2973, 0.2101, 0.2184, 0.3238, 0.3388, 0.2404, 0.2227, 0.2455, 0.3166,
                        0.1900, 0.2316, 0.2321, 0.2241, 0.2337, 0.1976, 0.3202, 0.2373, 0.3182,
                        0.3147, 0.2337, 0.2907, 0.2358, 0.2307, 0.3304, 0.2629, 0.1560, 0.2762,
                        0.2196, 0.2637, 0.2427, 0.2259, 0.2451, 0.2160, 0.2638, 0.2558, 0.2212,
                        0.2264, 0.2730, 0.2739, 0.3009, 0.2829, 0.2770, 0.3158, 0.2555, 0.3204,
                        0.3586, 0.2375, 0.2894, 0.2574, 0.2857, 0.2791, 0.2666, 0.2263, 0.1959,
                        0.1969, 0.2640, 0.2314, 0.2714, 0.2877, 0.2964, 0.2824, 0.2858, 0.1742,
                        0.2041, 0.2892, 0.3015, 0.2098, 0.2391, 0.3293, 0.2338, 0.2292, 0.2728,
                        0.2708, 0.2335, 0.2637, 0.2614, 0.2856, 0.2337, 0.2243, 0.2727, 0.2976,
                        0.2608, 0.2372, 0.2409, 0.2512, 0.2650, 0.2514, 0.2500, 0.1943, 0.3753,
                        0.2744, 0.2787, 0.3296, 0.3056, 0.2101, 0.2425, 0.2877, 0.1728, 0.2491,
                        0.3334, 0.1819, 0.2123, 0.3144, 0.2151])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.3030]), zero_point=tensor([76], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-23.067686080932617, max_val=15.413945198059082)
            )
          )
          (bn1): BatchNorm2d(
            509, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0777]), zero_point=tensor([62], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-4.835476875305176, max_val=5.038032054901123)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            509, 503, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0018, 0.0018, 0.0016, 0.0026, 0.0019, 0.0013, 0.0019, 0.0017, 0.0013,
                      0.0017, 0.0025, 0.0030, 0.0024, 0.0012, 0.0022, 0.0020, 0.0019, 0.0016,
                      0.0021, 0.0030, 0.0024, 0.0014, 0.0020, 0.0020, 0.0017, 0.0028, 0.0023,
                      0.0021, 0.0017, 0.0020, 0.0023, 0.0024, 0.0015, 0.0021, 0.0016, 0.0015,
                      0.0020, 0.0019, 0.0021, 0.0017, 0.0019, 0.0028, 0.0017, 0.0019, 0.0014,
                      0.0021, 0.0015, 0.0025, 0.0014, 0.0019, 0.0015, 0.0022, 0.0023, 0.0022,
                      0.0023, 0.0018, 0.0020, 0.0027, 0.0014, 0.0024, 0.0021, 0.0022, 0.0018,
                      0.0017, 0.0020, 0.0011, 0.0016, 0.0015, 0.0020, 0.0021, 0.0023, 0.0017,
                      0.0016, 0.0021, 0.0018, 0.0021, 0.0023, 0.0019, 0.0024, 0.0023, 0.0016,
                      0.0021, 0.0016, 0.0019, 0.0023, 0.0027, 0.0015, 0.0022, 0.0020, 0.0019,
                      0.0014, 0.0014, 0.0018, 0.0019, 0.0019, 0.0016, 0.0013, 0.0016, 0.0017,
                      0.0019, 0.0014, 0.0013, 0.0014, 0.0019, 0.0019, 0.0019, 0.0021, 0.0020,
                      0.0016, 0.0023, 0.0018, 0.0022, 0.0021, 0.0017, 0.0023, 0.0029, 0.0013,
                      0.0017, 0.0024, 0.0017, 0.0008, 0.0025, 0.0015, 0.0022, 0.0022, 0.0017,
                      0.0016, 0.0019, 0.0020, 0.0019, 0.0023, 0.0022, 0.0016, 0.0019, 0.0019,
                      0.0020, 0.0019, 0.0018, 0.0014, 0.0013, 0.0020, 0.0022, 0.0021, 0.0016,
                      0.0022, 0.0020, 0.0020, 0.0015, 0.0018, 0.0015, 0.0020, 0.0018, 0.0021,
                      0.0015, 0.0023, 0.0012, 0.0022, 0.0020, 0.0014, 0.0024, 0.0017, 0.0017,
                      0.0018, 0.0028, 0.0020, 0.0025, 0.0023, 0.0011, 0.0022, 0.0017, 0.0026,
                      0.0018, 0.0015, 0.0019, 0.0022, 0.0017, 0.0015, 0.0013, 0.0022, 0.0017,
                      0.0016, 0.0014, 0.0021, 0.0018, 0.0021, 0.0020, 0.0012, 0.0012, 0.0021,
                      0.0023, 0.0019, 0.0015, 0.0022, 0.0013, 0.0021, 0.0022, 0.0025, 0.0020,
                      0.0019, 0.0017, 0.0013, 0.0012, 0.0015, 0.0017, 0.0018, 0.0016, 0.0014,
                      0.0012, 0.0017, 0.0026, 0.0030, 0.0016, 0.0017, 0.0017, 0.0017, 0.0019,
                      0.0024, 0.0016, 0.0014, 0.0026, 0.0020, 0.0023, 0.0015, 0.0021, 0.0022,
                      0.0021, 0.0020, 0.0025, 0.0017, 0.0021, 0.0022, 0.0018, 0.0020, 0.0018,
                      0.0016, 0.0021, 0.0019, 0.0018, 0.0019, 0.0014, 0.0024, 0.0022, 0.0010,
                      0.0014, 0.0018, 0.0021, 0.0024, 0.0018, 0.0021, 0.0013, 0.0020, 0.0025,
                      0.0024, 0.0028, 0.0021, 0.0012, 0.0019, 0.0017, 0.0029, 0.0008, 0.0023,
                      0.0029, 0.0013, 0.0013, 0.0019, 0.0015, 0.0018, 0.0019, 0.0011, 0.0019,
                      0.0024, 0.0018, 0.0010, 0.0019, 0.0019, 0.0019, 0.0017, 0.0016, 0.0022,
                      0.0012, 0.0020, 0.0023, 0.0027, 0.0009, 0.0016, 0.0026, 0.0022, 0.0013,
                      0.0022, 0.0014, 0.0023, 0.0017, 0.0017, 0.0016, 0.0013, 0.0013, 0.0017,
                      0.0016, 0.0019, 0.0018, 0.0014, 0.0024, 0.0020, 0.0014, 0.0018, 0.0020,
                      0.0016, 0.0019, 0.0017, 0.0025, 0.0019, 0.0021, 0.0014, 0.0016, 0.0019,
                      0.0022, 0.0021, 0.0012, 0.0013, 0.0024, 0.0017, 0.0016, 0.0016, 0.0021,
                      0.0022, 0.0014, 0.0016, 0.0015, 0.0019, 0.0012, 0.0018, 0.0015, 0.0015,
                      0.0022, 0.0022, 0.0018, 0.0018, 0.0017, 0.0019, 0.0017, 0.0018, 0.0021,
                      0.0015, 0.0018, 0.0023, 0.0023, 0.0015, 0.0013, 0.0017, 0.0021, 0.0023,
                      0.0028, 0.0020, 0.0019, 0.0010, 0.0019, 0.0014, 0.0019, 0.0023, 0.0029,
                      0.0011, 0.0013, 0.0016, 0.0015, 0.0019, 0.0013, 0.0017, 0.0014, 0.0019,
                      0.0016, 0.0019, 0.0011, 0.0030, 0.0014, 0.0015, 0.0014, 0.0020, 0.0017,
                      0.0021, 0.0023, 0.0015, 0.0022, 0.0014, 0.0016, 0.0031, 0.0008, 0.0029,
                      0.0022, 0.0014, 0.0027, 0.0021, 0.0013, 0.0025, 0.0014, 0.0015, 0.0016,
                      0.0019, 0.0018, 0.0017, 0.0022, 0.0017, 0.0021, 0.0015, 0.0022, 0.0022,
                      0.0019, 0.0013, 0.0015, 0.0024, 0.0021, 0.0013, 0.0015, 0.0026, 0.0016,
                      0.0017, 0.0014, 0.0015, 0.0012, 0.0025, 0.0009, 0.0024, 0.0020, 0.0017,
                      0.0015, 0.0019, 0.0020, 0.0017, 0.0014, 0.0022, 0.0021, 0.0012, 0.0019,
                      0.0017, 0.0017, 0.0029, 0.0020, 0.0015, 0.0021, 0.0018, 0.0022, 0.0015,
                      0.0020, 0.0021, 0.0019, 0.0019, 0.0016, 0.0019, 0.0020, 0.0021, 0.0023,
                      0.0021, 0.0020, 0.0017, 0.0025, 0.0021, 0.0013, 0.0015, 0.0017, 0.0015,
                      0.0015, 0.0023, 0.0020, 0.0024, 0.0017, 0.0022, 0.0017, 0.0014, 0.0017,
                      0.0015, 0.0017, 0.0015, 0.0030, 0.0018, 0.0033, 0.0020, 0.0012, 0.0015,
                      0.0015, 0.0018, 0.0021, 0.0019, 0.0016, 0.0017, 0.0021, 0.0015, 0.0022,
                      0.0025, 0.0027, 0.0020, 0.0012, 0.0022, 0.0028, 0.0015, 0.0011, 0.0015,
                      0.0019, 0.0021, 0.0015, 0.0018, 0.0009, 0.0020, 0.0020, 0.0018]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.2027, -0.2299, -0.1908, -0.2364, -0.2346, -0.1313, -0.2478, -0.1937,
                        -0.1694, -0.1236, -0.3214, -0.2444, -0.2412, -0.1532, -0.2824, -0.2122,
                        -0.2190, -0.2102, -0.2028, -0.3838, -0.2385, -0.1449, -0.2502, -0.2304,
                        -0.2190, -0.3567, -0.2593, -0.1995, -0.2231, -0.2530, -0.2590, -0.3101,
                        -0.1907, -0.2670, -0.1941, -0.1469, -0.2605, -0.1611, -0.2665, -0.2205,
                        -0.1735, -0.3539, -0.1860, -0.2134, -0.1784, -0.2020, -0.1791, -0.3200,
                        -0.1796, -0.2444, -0.1718, -0.2823, -0.2940, -0.2863, -0.2134, -0.2120,
                        -0.1760, -0.2644, -0.1760, -0.1178, -0.2550, -0.2878, -0.2285, -0.1811,
                        -0.2128, -0.1462, -0.1532, -0.1982, -0.1912, -0.2636, -0.3008, -0.2162,
                        -0.2025, -0.2735, -0.1397, -0.2737, -0.2997, -0.2459, -0.3032, -0.2989,
                        -0.2074, -0.2420, -0.2070, -0.2376, -0.2548, -0.3468, -0.1797, -0.1843,
                        -0.2497, -0.2421, -0.1818, -0.1375, -0.1997, -0.2235, -0.1996, -0.1791,
                        -0.1611, -0.2079, -0.1052, -0.2205, -0.1768, -0.1666, -0.1590, -0.1892,
                        -0.2068, -0.2416, -0.2721, -0.2508, -0.1736, -0.1883, -0.2269, -0.2482,
                        -0.2348, -0.1965, -0.2145, -0.3186, -0.1707, -0.1993, -0.2659, -0.1615,
                        -0.1084, -0.3163, -0.1941, -0.2078, -0.2669, -0.1844, -0.2092, -0.2369,
                        -0.2529, -0.2039, -0.2244, -0.2185, -0.1720, -0.2090, -0.2422, -0.2332,
                        -0.2488, -0.1771, -0.1754, -0.0996, -0.2605, -0.2346, -0.2719, -0.1298,
                        -0.2757, -0.2564, -0.2610, -0.1415, -0.2273, -0.1952, -0.2578, -0.2175,
                        -0.2709, -0.1464, -0.2518, -0.1363, -0.2621, -0.2557, -0.1762, -0.2137,
                        -0.2139, -0.1567, -0.1716, -0.2765, -0.2563, -0.2614, -0.2211, -0.1451,
                        -0.2752, -0.2164, -0.3282, -0.2325, -0.1737, -0.2466, -0.2393, -0.1539,
                        -0.1663, -0.1701, -0.1594, -0.2199, -0.1764, -0.1600, -0.2152, -0.2248,
                        -0.2613, -0.2559, -0.1493, -0.1473, -0.2678, -0.2903, -0.2390, -0.1779,
                        -0.2849, -0.1714, -0.2627, -0.2877, -0.2377, -0.1973, -0.2457, -0.2137,
                        -0.1628, -0.1595, -0.1900, -0.2114, -0.2068, -0.1835, -0.1845, -0.1446,
                        -0.1759, -0.3373, -0.2229, -0.1919, -0.1972, -0.2161, -0.2092, -0.2492,
                        -0.2185, -0.2040, -0.1282, -0.3284, -0.2566, -0.3003, -0.1671, -0.2249,
                        -0.1805, -0.2296, -0.1529, -0.2799, -0.1954, -0.2682, -0.2820, -0.2336,
                        -0.2231, -0.2037, -0.2091, -0.2719, -0.2265, -0.2312, -0.2418, -0.1740,
                        -0.2443, -0.2782, -0.1258, -0.1764, -0.2134, -0.2355, -0.2596, -0.2322,
                        -0.2332, -0.1471, -0.2368, -0.3171, -0.2888, -0.2364, -0.1818, -0.1538,
                        -0.2407, -0.1870, -0.2597, -0.0901, -0.2882, -0.3730, -0.1513, -0.1588,
                        -0.1801, -0.1635, -0.2037, -0.2399, -0.1107, -0.2152, -0.2132, -0.2322,
                        -0.1305, -0.2313, -0.2392, -0.2461, -0.1976, -0.1213, -0.1490, -0.1484,
                        -0.2405, -0.2326, -0.3398, -0.1059, -0.1997, -0.3268, -0.2504, -0.1661,
                        -0.2799, -0.1751, -0.2578, -0.2211, -0.2172, -0.2008, -0.1610, -0.1608,
                        -0.1734, -0.2001, -0.2368, -0.2329, -0.1715, -0.3024, -0.1976, -0.1581,
                        -0.2273, -0.1954, -0.2057, -0.1798, -0.2178, -0.2260, -0.2496, -0.2697,
                        -0.1320, -0.1366, -0.2394, -0.2073, -0.2630, -0.1245, -0.1390, -0.3016,
                        -0.1993, -0.1924, -0.1946, -0.2067, -0.2581, -0.1169, -0.2093, -0.1688,
                        -0.2403, -0.1517, -0.1553, -0.1944, -0.1871, -0.2666, -0.2327, -0.2352,
                        -0.2273, -0.2134, -0.2438, -0.2226, -0.1506, -0.2748, -0.1740, -0.2316,
                        -0.2092, -0.2367, -0.1628, -0.1676, -0.2190, -0.1517, -0.2998, -0.3562,
                        -0.2618, -0.2468, -0.0956, -0.2385, -0.1662, -0.2374, -0.3002, -0.2873,
                        -0.1463, -0.1718, -0.2104, -0.1968, -0.2469, -0.1225, -0.1919, -0.1723,
                        -0.2388, -0.1401, -0.1779, -0.1180, -0.3778, -0.1634, -0.1742, -0.1654,
                        -0.2386, -0.2140, -0.2644, -0.2992, -0.1943, -0.2505, -0.1454, -0.1461,
                        -0.2150, -0.0989, -0.2445, -0.2372, -0.1802, -0.3457, -0.2384, -0.1630,
                        -0.3222, -0.1092, -0.1377, -0.1590, -0.1600, -0.2283, -0.1832, -0.1863,
                        -0.2184, -0.1876, -0.1875, -0.2765, -0.1914, -0.2417, -0.1611, -0.1306,
                        -0.3111, -0.2080, -0.1658, -0.1827, -0.2329, -0.2045, -0.2217, -0.1815,
                        -0.1847, -0.1560, -0.3195, -0.1097, -0.3135, -0.2437, -0.2154, -0.1915,
                        -0.2421, -0.2044, -0.2053, -0.1798, -0.2855, -0.2693, -0.1243, -0.1097,
                        -0.2186, -0.1770, -0.3761, -0.2609, -0.1897, -0.2699, -0.2302, -0.2815,
                        -0.1757, -0.2456, -0.2726, -0.2416, -0.2124, -0.2108, -0.1530, -0.2498,
                        -0.2638, -0.2961, -0.2204, -0.2188, -0.2225, -0.2463, -0.1817, -0.1271,
                        -0.1927, -0.2152, -0.1940, -0.1500, -0.2908, -0.2326, -0.3109, -0.2119,
                        -0.1662, -0.1403, -0.1762, -0.2145, -0.1925, -0.1589, -0.1959, -0.2270,
                        -0.1935, -0.3296, -0.2604, -0.1311, -0.1950, -0.1900, -0.2261, -0.2182,
                        -0.1648, -0.2078, -0.2173, -0.2651, -0.1245, -0.1603, -0.2540, -0.3438,
                        -0.2553, -0.1488, -0.2761, -0.3631, -0.1789, -0.1385, -0.1842, -0.2421,
                        -0.2713, -0.1683, -0.1820, -0.1144, -0.2150, -0.0922, -0.2260]), max_val=tensor([0.2325, 0.1972, 0.2094, 0.3278, 0.2470, 0.1660, 0.1671, 0.2126, 0.1530,
                        0.2113, 0.2288, 0.3845, 0.3026, 0.1397, 0.2324, 0.2560, 0.2421, 0.2058,
                        0.2610, 0.2419, 0.3045, 0.1828, 0.2507, 0.2489, 0.1613, 0.2181, 0.2973,
                        0.2707, 0.1805, 0.1862, 0.2905, 0.2297, 0.1765, 0.2631, 0.2054, 0.1914,
                        0.1891, 0.2433, 0.2038, 0.1875, 0.2387, 0.1932, 0.2220, 0.2383, 0.1772,
                        0.2697, 0.1891, 0.1780, 0.1616, 0.2233, 0.1861, 0.2377, 0.1990, 0.2796,
                        0.2900, 0.2285, 0.2489, 0.3434, 0.1796, 0.3048, 0.2712, 0.2424, 0.2110,
                        0.2174, 0.2495, 0.1242, 0.2055, 0.1505, 0.2497, 0.2363, 0.1891, 0.1926,
                        0.1481, 0.1786, 0.2328, 0.2468, 0.1608, 0.2460, 0.2148, 0.2282, 0.1650,
                        0.2669, 0.1570, 0.2213, 0.2913, 0.2903, 0.1858, 0.2848, 0.2061, 0.2387,
                        0.1097, 0.1836, 0.2277, 0.2382, 0.2373, 0.2027, 0.1191, 0.1523, 0.2218,
                        0.2373, 0.1595, 0.1430, 0.1730, 0.2416, 0.2472, 0.2081, 0.2227, 0.2428,
                        0.2016, 0.2959, 0.1551, 0.2796, 0.2671, 0.2147, 0.2929, 0.3661, 0.1054,
                        0.2168, 0.2993, 0.2180, 0.1028, 0.2254, 0.1544, 0.2855, 0.2848, 0.2160,
                        0.1894, 0.1775, 0.2033, 0.2395, 0.2929, 0.2740, 0.2035, 0.2419, 0.1695,
                        0.2580, 0.2071, 0.2265, 0.1388, 0.1651, 0.1987, 0.2819, 0.2483, 0.2056,
                        0.2787, 0.2095, 0.2257, 0.1859, 0.2156, 0.1575, 0.2115, 0.2304, 0.2501,
                        0.1845, 0.2962, 0.1466, 0.2781, 0.2110, 0.1785, 0.3008, 0.1944, 0.2128,
                        0.2223, 0.3613, 0.2023, 0.3207, 0.2892, 0.1336, 0.2359, 0.1807, 0.2574,
                        0.2104, 0.1953, 0.2132, 0.2848, 0.2215, 0.1885, 0.1471, 0.2823, 0.1949,
                        0.1975, 0.1741, 0.2690, 0.2314, 0.2658, 0.2371, 0.1454, 0.1179, 0.2394,
                        0.2498, 0.1895, 0.1952, 0.2775, 0.1566, 0.2217, 0.1832, 0.3140, 0.2528,
                        0.2028, 0.2073, 0.1547, 0.1358, 0.1617, 0.1740, 0.2231, 0.2039, 0.1738,
                        0.1509, 0.2117, 0.1825, 0.3773, 0.2079, 0.2216, 0.1988, 0.2126, 0.2236,
                        0.2992, 0.1509, 0.1748, 0.2382, 0.2308, 0.2616, 0.1936, 0.2622, 0.2781,
                        0.2722, 0.2531, 0.3112, 0.2196, 0.2501, 0.2774, 0.2233, 0.2543, 0.2230,
                        0.1925, 0.1844, 0.2384, 0.1916, 0.1920, 0.1821, 0.3009, 0.2249, 0.0933,
                        0.1705, 0.2227, 0.2634, 0.3026, 0.1988, 0.2723, 0.1595, 0.2533, 0.2436,
                        0.3101, 0.3507, 0.2604, 0.1448, 0.1736, 0.2135, 0.3649, 0.1014, 0.1759,
                        0.2038, 0.1592, 0.1666, 0.2461, 0.1902, 0.2306, 0.1719, 0.1457, 0.2451,
                        0.3028, 0.1443, 0.1214, 0.2362, 0.2431, 0.2043, 0.2148, 0.2016, 0.2741,
                        0.1474, 0.2572, 0.2882, 0.2620, 0.1198, 0.1964, 0.2590, 0.2760, 0.1353,
                        0.2610, 0.1219, 0.2875, 0.2066, 0.2037, 0.1747, 0.1161, 0.1644, 0.2152,
                        0.1854, 0.2372, 0.1737, 0.1726, 0.1990, 0.2565, 0.1756, 0.1874, 0.2521,
                        0.1883, 0.2427, 0.1762, 0.3139, 0.2460, 0.1955, 0.1726, 0.2083, 0.2401,
                        0.2839, 0.2167, 0.1545, 0.1706, 0.2291, 0.2215, 0.1972, 0.1979, 0.2620,
                        0.2788, 0.1769, 0.1834, 0.1877, 0.1985, 0.1285, 0.2250, 0.1872, 0.1176,
                        0.2817, 0.2777, 0.1577, 0.2344, 0.2104, 0.1615, 0.1868, 0.2266, 0.2497,
                        0.1923, 0.1870, 0.2899, 0.2891, 0.1849, 0.1448, 0.1995, 0.2721, 0.2232,
                        0.2723, 0.2360, 0.2370, 0.1313, 0.2386, 0.1786, 0.2455, 0.1977, 0.3655,
                        0.1410, 0.1417, 0.1454, 0.1418, 0.1797, 0.1659, 0.2120, 0.1762, 0.1870,
                        0.2080, 0.2412, 0.1385, 0.3092, 0.1759, 0.1888, 0.1719, 0.2538, 0.1907,
                        0.2138, 0.1655, 0.1935, 0.2785, 0.1718, 0.2025, 0.3934, 0.1065, 0.3666,
                        0.2839, 0.1705, 0.3049, 0.2617, 0.1266, 0.2490, 0.1778, 0.1933, 0.2060,
                        0.2363, 0.1632, 0.2136, 0.2803, 0.1802, 0.2639, 0.1474, 0.2180, 0.2787,
                        0.1781, 0.1246, 0.1961, 0.2197, 0.2625, 0.1328, 0.1854, 0.3249, 0.1604,
                        0.1999, 0.1135, 0.1921, 0.1575, 0.2984, 0.0933, 0.2016, 0.2585, 0.2161,
                        0.1602, 0.2476, 0.2479, 0.2134, 0.1582, 0.2796, 0.2188, 0.1478, 0.2391,
                        0.1546, 0.2109, 0.2468, 0.1840, 0.1456, 0.1859, 0.2102, 0.2387, 0.1886,
                        0.2504, 0.1966, 0.2172, 0.2391, 0.1675, 0.2452, 0.2031, 0.2423, 0.2694,
                        0.2662, 0.2601, 0.1688, 0.3128, 0.2676, 0.1696, 0.1733, 0.2082, 0.1766,
                        0.1943, 0.2706, 0.2546, 0.2651, 0.1734, 0.2852, 0.2203, 0.1572, 0.1967,
                        0.1566, 0.2210, 0.1956, 0.3803, 0.2345, 0.4248, 0.2166, 0.1467, 0.1883,
                        0.1856, 0.1989, 0.2645, 0.2445, 0.2064, 0.1910, 0.1696, 0.1924, 0.2759,
                        0.3134, 0.2012, 0.2142, 0.1507, 0.1815, 0.2941, 0.1895, 0.0993, 0.1907,
                        0.1751, 0.2045, 0.1956, 0.2229, 0.1027, 0.2483, 0.2553, 0.1589])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.4004]), zero_point=tensor([65], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-25.977060317993164, max_val=24.8764591217041)
            )
          )
          (bn2): BatchNorm2d(
            503, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0147]), zero_point=tensor([65], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.9553192257881165, max_val=0.9075891375541687)
            )
          )
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(
              247, 503, kernel_size=(1, 1), stride=(2, 2), bias=False
              (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0020, 0.0019, 0.0020, 0.0020, 0.0019, 0.0016, 0.0022, 0.0021, 0.0018,
                        0.0025, 0.0022, 0.0023, 0.0019, 0.0018, 0.0020, 0.0022, 0.0020, 0.0018,
                        0.0018, 0.0019, 0.0018, 0.0020, 0.0017, 0.0019, 0.0019, 0.0016, 0.0017,
                        0.0017, 0.0017, 0.0019, 0.0022, 0.0019, 0.0016, 0.0020, 0.0017, 0.0020,
                        0.0018, 0.0021, 0.0017, 0.0023, 0.0019, 0.0017, 0.0020, 0.0018, 0.0016,
                        0.0024, 0.0021, 0.0027, 0.0016, 0.0018, 0.0018, 0.0017, 0.0021, 0.0019,
                        0.0015, 0.0020, 0.0020, 0.0019, 0.0022, 0.0016, 0.0025, 0.0020, 0.0020,
                        0.0023, 0.0020, 0.0018, 0.0018, 0.0017, 0.0018, 0.0016, 0.0021, 0.0017,
                        0.0017, 0.0016, 0.0018, 0.0021, 0.0018, 0.0019, 0.0026, 0.0019, 0.0018,
                        0.0019, 0.0023, 0.0016, 0.0021, 0.0021, 0.0018, 0.0015, 0.0017, 0.0019,
                        0.0017, 0.0017, 0.0023, 0.0021, 0.0018, 0.0018, 0.0017, 0.0019, 0.0019,
                        0.0018, 0.0022, 0.0016, 0.0020, 0.0017, 0.0024, 0.0020, 0.0026, 0.0019,
                        0.0020, 0.0019, 0.0016, 0.0023, 0.0021, 0.0018, 0.0021, 0.0024, 0.0020,
                        0.0032, 0.0018, 0.0017, 0.0019, 0.0026, 0.0017, 0.0018, 0.0018, 0.0020,
                        0.0021, 0.0018, 0.0017, 0.0018, 0.0021, 0.0025, 0.0018, 0.0016, 0.0018,
                        0.0015, 0.0022, 0.0019, 0.0023, 0.0014, 0.0020, 0.0020, 0.0021, 0.0018,
                        0.0019, 0.0017, 0.0018, 0.0015, 0.0022, 0.0020, 0.0018, 0.0019, 0.0018,
                        0.0018, 0.0021, 0.0019, 0.0018, 0.0025, 0.0018, 0.0022, 0.0025, 0.0022,
                        0.0020, 0.0017, 0.0017, 0.0018, 0.0026, 0.0015, 0.0023, 0.0028, 0.0021,
                        0.0017, 0.0019, 0.0024, 0.0016, 0.0019, 0.0017, 0.0019, 0.0022, 0.0015,
                        0.0018, 0.0021, 0.0019, 0.0019, 0.0019, 0.0021, 0.0018, 0.0021, 0.0019,
                        0.0018, 0.0017, 0.0018, 0.0022, 0.0019, 0.0017, 0.0023, 0.0022, 0.0018,
                        0.0018, 0.0023, 0.0019, 0.0021, 0.0016, 0.0018, 0.0022, 0.0017, 0.0017,
                        0.0016, 0.0017, 0.0021, 0.0023, 0.0017, 0.0022, 0.0018, 0.0016, 0.0017,
                        0.0023, 0.0020, 0.0018, 0.0019, 0.0019, 0.0019, 0.0017, 0.0022, 0.0022,
                        0.0016, 0.0022, 0.0016, 0.0019, 0.0017, 0.0021, 0.0018, 0.0019, 0.0017,
                        0.0019, 0.0022, 0.0019, 0.0020, 0.0019, 0.0017, 0.0018, 0.0022, 0.0020,
                        0.0017, 0.0017, 0.0019, 0.0020, 0.0019, 0.0019, 0.0015, 0.0019, 0.0022,
                        0.0020, 0.0023, 0.0019, 0.0019, 0.0017, 0.0024, 0.0017, 0.0014, 0.0020,
                        0.0018, 0.0019, 0.0021, 0.0015, 0.0022, 0.0021, 0.0021, 0.0020, 0.0020,
                        0.0019, 0.0017, 0.0015, 0.0016, 0.0018, 0.0022, 0.0015, 0.0019, 0.0020,
                        0.0022, 0.0020, 0.0023, 0.0021, 0.0019, 0.0022, 0.0018, 0.0020, 0.0018,
                        0.0020, 0.0016, 0.0017, 0.0020, 0.0020, 0.0019, 0.0014, 0.0022, 0.0016,
                        0.0015, 0.0019, 0.0019, 0.0018, 0.0023, 0.0019, 0.0018, 0.0020, 0.0019,
                        0.0016, 0.0022, 0.0019, 0.0019, 0.0020, 0.0016, 0.0017, 0.0022, 0.0020,
                        0.0019, 0.0020, 0.0014, 0.0016, 0.0021, 0.0017, 0.0017, 0.0016, 0.0020,
                        0.0020, 0.0018, 0.0020, 0.0024, 0.0015, 0.0017, 0.0017, 0.0016, 0.0014,
                        0.0021, 0.0022, 0.0016, 0.0023, 0.0017, 0.0017, 0.0022, 0.0018, 0.0022,
                        0.0022, 0.0023, 0.0024, 0.0023, 0.0017, 0.0015, 0.0016, 0.0019, 0.0018,
                        0.0020, 0.0022, 0.0018, 0.0018, 0.0015, 0.0019, 0.0017, 0.0023, 0.0018,
                        0.0014, 0.0019, 0.0023, 0.0019, 0.0021, 0.0017, 0.0022, 0.0019, 0.0019,
                        0.0028, 0.0018, 0.0014, 0.0020, 0.0016, 0.0017, 0.0019, 0.0024, 0.0018,
                        0.0022, 0.0017, 0.0023, 0.0016, 0.0015, 0.0017, 0.0020, 0.0018, 0.0023,
                        0.0017, 0.0021, 0.0019, 0.0021, 0.0016, 0.0018, 0.0020, 0.0018, 0.0018,
                        0.0016, 0.0017, 0.0020, 0.0019, 0.0023, 0.0024, 0.0018, 0.0019, 0.0021,
                        0.0018, 0.0017, 0.0015, 0.0020, 0.0020, 0.0016, 0.0016, 0.0018, 0.0023,
                        0.0018, 0.0016, 0.0022, 0.0016, 0.0030, 0.0017, 0.0016, 0.0018, 0.0017,
                        0.0014, 0.0018, 0.0018, 0.0020, 0.0021, 0.0018, 0.0019, 0.0016, 0.0022,
                        0.0015, 0.0020, 0.0020, 0.0019, 0.0017, 0.0020, 0.0020, 0.0021, 0.0020,
                        0.0022, 0.0019, 0.0020, 0.0020, 0.0019, 0.0020, 0.0024, 0.0017, 0.0016,
                        0.0020, 0.0021, 0.0018, 0.0020, 0.0021, 0.0018, 0.0016, 0.0018, 0.0017,
                        0.0018, 0.0021, 0.0021, 0.0022, 0.0019, 0.0022, 0.0021, 0.0019, 0.0017,
                        0.0018, 0.0024, 0.0016, 0.0023, 0.0016, 0.0023, 0.0022, 0.0019, 0.0019,
                        0.0023, 0.0021, 0.0018, 0.0024, 0.0018, 0.0020, 0.0019, 0.0018, 0.0021,
                        0.0023, 0.0019, 0.0022, 0.0017, 0.0019, 0.0023, 0.0017, 0.0019, 0.0018,
                        0.0017, 0.0019, 0.0019, 0.0018, 0.0015, 0.0021, 0.0016, 0.0022]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
                (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                  min_val=tensor([-0.2262, -0.2443, -0.1880, -0.2542, -0.2443, -0.1644, -0.2410, -0.2128,
                          -0.2246, -0.3229, -0.2246, -0.2929, -0.2422, -0.2293, -0.2204, -0.2634,
                          -0.2515, -0.2151, -0.2259, -0.2387, -0.2301, -0.2615, -0.2115, -0.2371,
                          -0.2386, -0.2030, -0.1857, -0.2114, -0.2128, -0.1980, -0.2799, -0.2418,
                          -0.2012, -0.2505, -0.2221, -0.2541, -0.2245, -0.2691, -0.1874, -0.2888,
                          -0.2385, -0.1998, -0.1974, -0.2256, -0.2011, -0.3083, -0.2684, -0.3425,
                          -0.2091, -0.2290, -0.1965, -0.2099, -0.2745, -0.2456, -0.1936, -0.2618,
                          -0.2564, -0.2376, -0.2867, -0.2110, -0.3170, -0.2613, -0.2572, -0.2926,
                          -0.2513, -0.2237, -0.2282, -0.2114, -0.2302, -0.2105, -0.2741, -0.1939,
                          -0.2211, -0.2009, -0.2293, -0.2059, -0.2220, -0.2397, -0.3293, -0.2417,
                          -0.2241, -0.2402, -0.2995, -0.1936, -0.2709, -0.2734, -0.2018, -0.1979,
                          -0.2230, -0.2253, -0.2159, -0.2150, -0.2431, -0.2460, -0.2298, -0.2292,
                          -0.2146, -0.2401, -0.2027, -0.2278, -0.2338, -0.2041, -0.2536, -0.2134,
                          -0.3096, -0.2090, -0.3275, -0.2073, -0.2533, -0.2456, -0.2045, -0.2396,
                          -0.2675, -0.2351, -0.2725, -0.3096, -0.2521, -0.4071, -0.2334, -0.2026,
                          -0.2477, -0.3317, -0.2113, -0.2261, -0.2307, -0.2547, -0.2124, -0.2247,
                          -0.2166, -0.2284, -0.2746, -0.3162, -0.2335, -0.2012, -0.2114, -0.1949,
                          -0.2873, -0.2396, -0.2946, -0.1841, -0.2517, -0.2535, -0.2148, -0.2250,
                          -0.2399, -0.2239, -0.1994, -0.1921, -0.2812, -0.2589, -0.1976, -0.2480,
                          -0.2279, -0.2340, -0.2736, -0.2447, -0.2325, -0.3203, -0.2338, -0.2757,
                          -0.3148, -0.2780, -0.2512, -0.2148, -0.1983, -0.2276, -0.3357, -0.1894,
                          -0.2957, -0.3574, -0.2681, -0.2165, -0.2383, -0.3075, -0.1919, -0.2448,
                          -0.2228, -0.2412, -0.2657, -0.1716, -0.2353, -0.2655, -0.2369, -0.2435,
                          -0.2480, -0.2373, -0.2335, -0.2716, -0.2138, -0.2251, -0.2214, -0.2265,
                          -0.2575, -0.2400, -0.2116, -0.2941, -0.2803, -0.2350, -0.2139, -0.2950,
                          -0.2053, -0.2636, -0.2030, -0.2351, -0.2768, -0.2143, -0.2207, -0.2040,
                          -0.2151, -0.2735, -0.2197, -0.2161, -0.2818, -0.2192, -0.2040, -0.2054,
                          -0.2883, -0.2582, -0.1663, -0.2430, -0.2073, -0.2417, -0.2158, -0.2861,
                          -0.2805, -0.2094, -0.2017, -0.1922, -0.2312, -0.1898, -0.2701, -0.2312,
                          -0.2451, -0.2171, -0.2373, -0.2410, -0.2258, -0.2565, -0.2103, -0.2209,
                          -0.2328, -0.2755, -0.2513, -0.2117, -0.2154, -0.2234, -0.2258, -0.2443,
                          -0.2493, -0.1859, -0.2373, -0.2798, -0.1997, -0.2972, -0.2427, -0.2399,
                          -0.2199, -0.3050, -0.2237, -0.1809, -0.2592, -0.2282, -0.2422, -0.2179,
                          -0.1705, -0.2833, -0.2627, -0.2695, -0.2501, -0.2595, -0.2091, -0.2155,
                          -0.1959, -0.2032, -0.2341, -0.2759, -0.1905, -0.2388, -0.2411, -0.2845,
                          -0.2160, -0.2900, -0.2577, -0.2400, -0.2836, -0.2319, -0.2453, -0.2363,
                          -0.2564, -0.2097, -0.1916, -0.2542, -0.2594, -0.2172, -0.1768, -0.2756,
                          -0.2081, -0.1829, -0.2398, -0.2382, -0.2270, -0.3004, -0.2449, -0.2265,
                          -0.2593, -0.2404, -0.2106, -0.2783, -0.2487, -0.2467, -0.2611, -0.2019,
                          -0.2051, -0.1880, -0.2500, -0.2299, -0.2280, -0.1742, -0.1989, -0.2675,
                          -0.2134, -0.1862, -0.2078, -0.2586, -0.1885, -0.2334, -0.2613, -0.3060,
                          -0.1961, -0.2054, -0.2092, -0.2092, -0.1692, -0.1973, -0.2763, -0.2045,
                          -0.2974, -0.2160, -0.2206, -0.2225, -0.2254, -0.2801, -0.2853, -0.2914,
                          -0.3040, -0.2890, -0.2197, -0.1928, -0.2019, -0.2464, -0.2341, -0.2533,
                          -0.2763, -0.2314, -0.2312, -0.1921, -0.2465, -0.2114, -0.2942, -0.2345,
                          -0.1639, -0.2493, -0.2966, -0.1793, -0.2717, -0.2222, -0.2828, -0.2399,
                          -0.2493, -0.3583, -0.2126, -0.1759, -0.2598, -0.2094, -0.2145, -0.2421,
                          -0.3085, -0.2308, -0.2009, -0.2165, -0.2998, -0.2055, -0.1876, -0.2222,
                          -0.2614, -0.1462, -0.2230, -0.2195, -0.2730, -0.2428, -0.2691, -0.2075,
                          -0.2311, -0.2541, -0.2261, -0.2139, -0.2024, -0.2177, -0.2595, -0.2100,
                          -0.2953, -0.3111, -0.2289, -0.2398, -0.2627, -0.2306, -0.2071, -0.1913,
                          -0.1965, -0.2485, -0.1996, -0.1909, -0.2297, -0.2940, -0.2274, -0.2112,
                          -0.2823, -0.1864, -0.2190, -0.2214, -0.2054, -0.2318, -0.2194, -0.1730,
                          -0.2205, -0.2241, -0.2597, -0.2686, -0.2276, -0.2425, -0.1948, -0.2778,
                          -0.1951, -0.2500, -0.2606, -0.2450, -0.1710, -0.2525, -0.2535, -0.2716,
                          -0.1967, -0.2843, -0.1728, -0.2530, -0.2285, -0.2372, -0.2540, -0.3107,
                          -0.2239, -0.2022, -0.1713, -0.2276, -0.2314, -0.2536, -0.2674, -0.2351,
                          -0.1943, -0.2240, -0.2151, -0.2112, -0.2666, -0.2728, -0.2367, -0.2302,
                          -0.2833, -0.2040, -0.2394, -0.2225, -0.2332, -0.3049, -0.2104, -0.2134,
                          -0.2019, -0.2981, -0.2815, -0.2403, -0.2399, -0.2253, -0.2699, -0.2340,
                          -0.3079, -0.1722, -0.2584, -0.2440, -0.2264, -0.2647, -0.2920, -0.2377,
                          -0.2762, -0.1671, -0.2410, -0.2897, -0.2131, -0.2375, -0.2297, -0.1964,
                          -0.1910, -0.2385, -0.2270, -0.1751, -0.2651, -0.2019, -0.2769]), max_val=tensor([0.2527, 0.1807, 0.2495, 0.1648, 0.2357, 0.1986, 0.2795, 0.2608, 0.2114,
                          0.2222, 0.2733, 0.2477, 0.1603, 0.2051, 0.2515, 0.2852, 0.1753, 0.2317,
                          0.1922, 0.2383, 0.1878, 0.2018, 0.1742, 0.1368, 0.1686, 0.2032, 0.2165,
                          0.2121, 0.2015, 0.2454, 0.2732, 0.2327, 0.1822, 0.2342, 0.1918, 0.1981,
                          0.2012, 0.2607, 0.2110, 0.2912, 0.1773, 0.2146, 0.2564, 0.1682, 0.2003,
                          0.1900, 0.1962, 0.1958, 0.1784, 0.2275, 0.2318, 0.2097, 0.1826, 0.1840,
                          0.1904, 0.1721, 0.1492, 0.2192, 0.1545, 0.1716, 0.2116, 0.2106, 0.1851,
                          0.1879, 0.1698, 0.2269, 0.2163, 0.1471, 0.1977, 0.1692, 0.2382, 0.2111,
                          0.2203, 0.1940, 0.1279, 0.2665, 0.2223, 0.2009, 0.1699, 0.1786, 0.2110,
                          0.2304, 0.2873, 0.1972, 0.2202, 0.2275, 0.2273, 0.1651, 0.1922, 0.2428,
                          0.1757, 0.1765, 0.2971, 0.2730, 0.1641, 0.1679, 0.1731, 0.2060, 0.2376,
                          0.1682, 0.2735, 0.1758, 0.1883, 0.1702, 0.1983, 0.2554, 0.2542, 0.2370,
                          0.1943, 0.1855, 0.1790, 0.2866, 0.1958, 0.1461, 0.1579, 0.2123, 0.2166,
                          0.2881, 0.1847, 0.2179, 0.1852, 0.1992, 0.1986, 0.2269, 0.1957, 0.1890,
                          0.2621, 0.1648, 0.1943, 0.2078, 0.2486, 0.1697, 0.2084, 0.2085, 0.2253,
                          0.1883, 0.2242, 0.2350, 0.2012, 0.1655, 0.1754, 0.2368, 0.2624, 0.1379,
                          0.2306, 0.2023, 0.2330, 0.1796, 0.2653, 0.2147, 0.2251, 0.2370, 0.2291,
                          0.1924, 0.2278, 0.2104, 0.2038, 0.1912, 0.2059, 0.2753, 0.1680, 0.2178,
                          0.2034, 0.1744, 0.2183, 0.2097, 0.2489, 0.1863, 0.1927, 0.2108, 0.1795,
                          0.1513, 0.2140, 0.1992, 0.2050, 0.2406, 0.1782, 0.2139, 0.2779, 0.1881,
                          0.1770, 0.1922, 0.2240, 0.2422, 0.2130, 0.2690, 0.1471, 0.1954, 0.2404,
                          0.1892, 0.2079, 0.1965, 0.2794, 0.1764, 0.1780, 0.1911, 0.1943, 0.1857,
                          0.2283, 0.1890, 0.2362, 0.2136, 0.1894, 0.1596, 0.1897, 0.1955, 0.1824,
                          0.1676, 0.1885, 0.1972, 0.2904, 0.1750, 0.1987, 0.2312, 0.1587, 0.2146,
                          0.1877, 0.1682, 0.2302, 0.1803, 0.2392, 0.2404, 0.1510, 0.1853, 0.2038,
                          0.1861, 0.2751, 0.2070, 0.2378, 0.2100, 0.2387, 0.2102, 0.2310, 0.1961,
                          0.2011, 0.2820, 0.2418, 0.1804, 0.2415, 0.1906, 0.2006, 0.1630, 0.1971,
                          0.2010, 0.1934, 0.2364, 0.2490, 0.1825, 0.1533, 0.1424, 0.1892, 0.2085,
                          0.2509, 0.2003, 0.1692, 0.1732, 0.1900, 0.2565, 0.1649, 0.1611, 0.1750,
                          0.1602, 0.2141, 0.2715, 0.1921, 0.2002, 0.1641, 0.1716, 0.1778, 0.2038,
                          0.2469, 0.1751, 0.1733, 0.1851, 0.2167, 0.1737, 0.1664, 0.2024, 0.2602,
                          0.2266, 0.2521, 0.2662, 0.2718, 0.2263, 0.2230, 0.2088, 0.2486, 0.2046,
                          0.2208, 0.1607, 0.2144, 0.2150, 0.1876, 0.2367, 0.1391, 0.2031, 0.1739,
                          0.1916, 0.1609, 0.1950, 0.1455, 0.2531, 0.2224, 0.2345, 0.1554, 0.2014,
                          0.1612, 0.1816, 0.2042, 0.1905, 0.2441, 0.1952, 0.2180, 0.2779, 0.1854,
                          0.2370, 0.2540, 0.1451, 0.1762, 0.2563, 0.1838, 0.2177, 0.1504, 0.1672,
                          0.2542, 0.1630, 0.2216, 0.2057, 0.1826, 0.2119, 0.2101, 0.1941, 0.1797,
                          0.2706, 0.1750, 0.1880, 0.2226, 0.1812, 0.1717, 0.2838, 0.1830, 0.2197,
                          0.1737, 0.1825, 0.2481, 0.1953, 0.1807, 0.1708, 0.1946, 0.1859, 0.2110,
                          0.2417, 0.1730, 0.2243, 0.2270, 0.1900, 0.1913, 0.2054, 0.1959, 0.1920,
                          0.1813, 0.2287, 0.2002, 0.2454, 0.1776, 0.1494, 0.1506, 0.2334, 0.2463,
                          0.2578, 0.2285, 0.1773, 0.2120, 0.1769, 0.2178, 0.2144, 0.2591, 0.1757,
                          0.2768, 0.2021, 0.2372, 0.1772, 0.1452, 0.1823, 0.2050, 0.2245, 0.2902,
                          0.2010, 0.1425, 0.1748, 0.2456, 0.1814, 0.2151, 0.1873, 0.2277, 0.2311,
                          0.1902, 0.1673, 0.2347, 0.2407, 0.2332, 0.2068, 0.1589, 0.1352, 0.1894,
                          0.2086, 0.2100, 0.1607, 0.2498, 0.2595, 0.1669, 0.1971, 0.2217, 0.2307,
                          0.1788, 0.1791, 0.1889, 0.2060, 0.3848, 0.1940, 0.2016, 0.1952, 0.1796,
                          0.1477, 0.2285, 0.1853, 0.1907, 0.2139, 0.1760, 0.1873, 0.2069, 0.1793,
                          0.1883, 0.2101, 0.2455, 0.2353, 0.2145, 0.1975, 0.1763, 0.2100, 0.2486,
                          0.1992, 0.2379, 0.2427, 0.2494, 0.1823, 0.1794, 0.1839, 0.1928, 0.1970,
                          0.2587, 0.2673, 0.1795, 0.2127, 0.2231, 0.1426, 0.1992, 0.2223, 0.1999,
                          0.2260, 0.2524, 0.1618, 0.2797, 0.2362, 0.1872, 0.2724, 0.2450, 0.1955,
                          0.1878, 0.1859, 0.1883, 0.2930, 0.1771, 0.2440, 0.1955, 0.1947, 0.2161,
                          0.2894, 0.1845, 0.2253, 0.1690, 0.2294, 0.2205, 0.2422, 0.1497, 0.2379,
                          0.2368, 0.2421, 0.2699, 0.2215, 0.2397, 0.2295, 0.1777, 0.1950, 0.2121,
                          0.2182, 0.2365, 0.2036, 0.1401, 0.1958, 0.2133, 0.1918, 0.2420])
                )
              )
              (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0907]), zero_point=tensor([70], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=-6.34628963470459, max_val=5.166379451751709)
              )
            )
            (1): BatchNorm2d(
              503, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (activation_post_process): FusedMovingAvgObsFakeQuantize(
                fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0667]), zero_point=tensor([65], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
                (activation_post_process): MovingAverageMinMaxObserver(min_val=-4.360299587249756, max_val=4.108701229095459)
              )
            )
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(
            503, 509, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0008, 0.0007, 0.0008, 0.0008, 0.0007, 0.0011, 0.0013, 0.0008, 0.0006,
                      0.0006, 0.0007, 0.0007, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0007,
                      0.0007, 0.0008, 0.0007, 0.0006, 0.0009, 0.0007, 0.0007, 0.0007, 0.0007,
                      0.0007, 0.0008, 0.0013, 0.0007, 0.0006, 0.0007, 0.0011, 0.0006, 0.0006,
                      0.0007, 0.0009, 0.0006, 0.0006, 0.0009, 0.0008, 0.0007, 0.0009, 0.0007,
                      0.0007, 0.0009, 0.0014, 0.0013, 0.0007, 0.0006, 0.0006, 0.0008, 0.0007,
                      0.0006, 0.0011, 0.0007, 0.0007, 0.0007, 0.0006, 0.0008, 0.0006, 0.0009,
                      0.0006, 0.0006, 0.0018, 0.0008, 0.0006, 0.0012, 0.0013, 0.0008, 0.0007,
                      0.0013, 0.0007, 0.0013, 0.0011, 0.0006, 0.0010, 0.0008, 0.0007, 0.0008,
                      0.0007, 0.0006, 0.0011, 0.0006, 0.0007, 0.0007, 0.0007, 0.0006, 0.0007,
                      0.0007, 0.0009, 0.0007, 0.0007, 0.0009, 0.0008, 0.0006, 0.0008, 0.0006,
                      0.0006, 0.0007, 0.0006, 0.0007, 0.0017, 0.0007, 0.0007, 0.0007, 0.0007,
                      0.0010, 0.0007, 0.0010, 0.0008, 0.0007, 0.0007, 0.0006, 0.0011, 0.0011,
                      0.0006, 0.0008, 0.0008, 0.0007, 0.0009, 0.0006, 0.0007, 0.0008, 0.0008,
                      0.0007, 0.0012, 0.0009, 0.0009, 0.0008, 0.0008, 0.0007, 0.0008, 0.0007,
                      0.0007, 0.0008, 0.0008, 0.0007, 0.0011, 0.0006, 0.0006, 0.0007, 0.0006,
                      0.0007, 0.0010, 0.0011, 0.0007, 0.0011, 0.0010, 0.0007, 0.0006, 0.0007,
                      0.0009, 0.0007, 0.0007, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007,
                      0.0009, 0.0010, 0.0006, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
                      0.0012, 0.0009, 0.0006, 0.0007, 0.0006, 0.0008, 0.0008, 0.0013, 0.0010,
                      0.0008, 0.0010, 0.0007, 0.0012, 0.0007, 0.0007, 0.0007, 0.0007, 0.0008,
                      0.0007, 0.0006, 0.0007, 0.0007, 0.0006, 0.0006, 0.0012, 0.0010, 0.0007,
                      0.0008, 0.0007, 0.0009, 0.0007, 0.0009, 0.0011, 0.0010, 0.0008, 0.0011,
                      0.0009, 0.0007, 0.0007, 0.0013, 0.0009, 0.0008, 0.0006, 0.0007, 0.0012,
                      0.0006, 0.0006, 0.0014, 0.0009, 0.0007, 0.0008, 0.0008, 0.0011, 0.0008,
                      0.0007, 0.0010, 0.0008, 0.0008, 0.0015, 0.0011, 0.0007, 0.0008, 0.0012,
                      0.0008, 0.0007, 0.0011, 0.0012, 0.0008, 0.0006, 0.0006, 0.0007, 0.0007,
                      0.0014, 0.0008, 0.0007, 0.0008, 0.0007, 0.0012, 0.0008, 0.0007, 0.0008,
                      0.0007, 0.0007, 0.0006, 0.0007, 0.0007, 0.0006, 0.0008, 0.0007, 0.0007,
                      0.0006, 0.0008, 0.0007, 0.0007, 0.0014, 0.0006, 0.0011, 0.0006, 0.0006,
                      0.0006, 0.0010, 0.0009, 0.0006, 0.0008, 0.0009, 0.0007, 0.0007, 0.0007,
                      0.0016, 0.0006, 0.0011, 0.0006, 0.0010, 0.0008, 0.0007, 0.0008, 0.0007,
                      0.0012, 0.0008, 0.0016, 0.0006, 0.0008, 0.0006, 0.0010, 0.0006, 0.0007,
                      0.0010, 0.0008, 0.0008, 0.0006, 0.0009, 0.0008, 0.0007, 0.0007, 0.0007,
                      0.0016, 0.0006, 0.0007, 0.0006, 0.0007, 0.0008, 0.0007, 0.0006, 0.0008,
                      0.0008, 0.0007, 0.0008, 0.0006, 0.0009, 0.0009, 0.0009, 0.0014, 0.0009,
                      0.0008, 0.0006, 0.0006, 0.0010, 0.0007, 0.0006, 0.0021, 0.0007, 0.0008,
                      0.0008, 0.0007, 0.0009, 0.0006, 0.0007, 0.0008, 0.0010, 0.0007, 0.0009,
                      0.0007, 0.0007, 0.0009, 0.0007, 0.0007, 0.0009, 0.0012, 0.0007, 0.0007,
                      0.0009, 0.0009, 0.0009, 0.0006, 0.0008, 0.0007, 0.0007, 0.0011, 0.0007,
                      0.0007, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0007, 0.0010, 0.0008,
                      0.0007, 0.0008, 0.0009, 0.0008, 0.0007, 0.0008, 0.0009, 0.0007, 0.0007,
                      0.0006, 0.0007, 0.0007, 0.0007, 0.0013, 0.0006, 0.0009, 0.0007, 0.0008,
                      0.0010, 0.0010, 0.0008, 0.0007, 0.0006, 0.0008, 0.0007, 0.0006, 0.0007,
                      0.0009, 0.0008, 0.0009, 0.0010, 0.0006, 0.0008, 0.0007, 0.0007, 0.0007,
                      0.0012, 0.0006, 0.0006, 0.0009, 0.0006, 0.0014, 0.0006, 0.0007, 0.0009,
                      0.0006, 0.0006, 0.0010, 0.0013, 0.0011, 0.0008, 0.0014, 0.0011, 0.0007,
                      0.0008, 0.0007, 0.0006, 0.0008, 0.0007, 0.0008, 0.0011, 0.0009, 0.0010,
                      0.0007, 0.0007, 0.0007, 0.0008, 0.0008, 0.0007, 0.0008, 0.0007, 0.0009,
                      0.0009, 0.0009, 0.0007, 0.0006, 0.0007, 0.0008, 0.0006, 0.0008, 0.0009,
                      0.0006, 0.0007, 0.0011, 0.0010, 0.0008, 0.0007, 0.0012, 0.0007, 0.0007,
                      0.0006, 0.0006, 0.0007, 0.0014, 0.0007, 0.0009, 0.0009, 0.0007, 0.0007,
                      0.0008, 0.0009, 0.0009, 0.0007, 0.0008, 0.0007, 0.0008, 0.0009, 0.0010,
                      0.0007, 0.0007, 0.0007, 0.0008, 0.0007, 0.0007, 0.0006, 0.0007, 0.0006,
                      0.0008, 0.0007, 0.0007, 0.0010, 0.0008, 0.0007, 0.0011, 0.0006, 0.0006,
                      0.0006, 0.0008, 0.0006, 0.0007, 0.0012, 0.0008, 0.0009, 0.0007, 0.0007,
                      0.0010, 0.0006, 0.0010, 0.0008, 0.0008]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.1079, -0.0846, -0.1060, -0.1074, -0.0843, -0.1385, -0.1601, -0.1063,
                        -0.0757, -0.0776, -0.0933, -0.0896, -0.0917, -0.1034, -0.0890, -0.0873,
                        -0.0754, -0.0916, -0.0808, -0.0998, -0.0785, -0.0679, -0.1101, -0.0834,
                        -0.0900, -0.0821, -0.0747, -0.0835, -0.1074, -0.1699, -0.0922, -0.0831,
                        -0.0901, -0.1415, -0.0784, -0.0757, -0.0924, -0.1121, -0.0784, -0.0776,
                        -0.1106, -0.1057, -0.0850, -0.1109, -0.0952, -0.0852, -0.1193, -0.1746,
                        -0.1704, -0.0886, -0.0826, -0.0808, -0.1047, -0.0899, -0.0809, -0.1419,
                        -0.0869, -0.0866, -0.0924, -0.0768, -0.1019, -0.0794, -0.1135, -0.0746,
                        -0.0715, -0.2242, -0.1078, -0.0760, -0.1483, -0.1128, -0.1033, -0.0922,
                        -0.1630, -0.0881, -0.1621, -0.1411, -0.0811, -0.1293, -0.0996, -0.0735,
                        -0.0773, -0.0924, -0.0743, -0.1450, -0.0735, -0.0844, -0.0832, -0.0837,
                        -0.0768, -0.0854, -0.0863, -0.1212, -0.0878, -0.0844, -0.1131, -0.1027,
                        -0.0771, -0.0749, -0.0830, -0.0817, -0.0954, -0.0795, -0.0902, -0.0982,
                        -0.0852, -0.0844, -0.0855, -0.0818, -0.1336, -0.0789, -0.1292, -0.0736,
                        -0.0948, -0.0856, -0.0813, -0.1399, -0.1405, -0.0765, -0.1009, -0.1063,
                        -0.0842, -0.1152, -0.0675, -0.0849, -0.1047, -0.1011, -0.0884, -0.1513,
                        -0.1149, -0.1209, -0.0862, -0.0978, -0.0867, -0.0825, -0.0852, -0.0869,
                        -0.0986, -0.1048, -0.0829, -0.1448, -0.0823, -0.0767, -0.0942, -0.0710,
                        -0.0910, -0.1154, -0.1467, -0.0853, -0.1359, -0.1340, -0.0868, -0.0763,
                        -0.0916, -0.1141, -0.0836, -0.0765, -0.1006, -0.0897, -0.1049, -0.0957,
                        -0.0904, -0.0731, -0.1096, -0.1256, -0.0714, -0.0964, -0.0846, -0.0890,
                        -0.0865, -0.0922, -0.0770, -0.1575, -0.1162, -0.0791, -0.0865, -0.0721,
                        -0.0976, -0.0980, -0.1621, -0.1328, -0.0967, -0.0914, -0.0859, -0.1480,
                        -0.0947, -0.0841, -0.0770, -0.0910, -0.0978, -0.0754, -0.0823, -0.0858,
                        -0.0845, -0.0760, -0.0754, -0.1575, -0.1221, -0.0949, -0.0900, -0.0864,
                        -0.1201, -0.0943, -0.1176, -0.1465, -0.0899, -0.0967, -0.1441, -0.1166,
                        -0.0891, -0.0940, -0.1696, -0.1096, -0.0840, -0.0827, -0.0851, -0.1582,
                        -0.0802, -0.0777, -0.1820, -0.1104, -0.0850, -0.1015, -0.1059, -0.1431,
                        -0.0757, -0.0926, -0.1239, -0.0993, -0.0865, -0.1868, -0.1451, -0.0845,
                        -0.1081, -0.1527, -0.0975, -0.0870, -0.1423, -0.1591, -0.1038, -0.0783,
                        -0.0821, -0.0913, -0.0861, -0.1486, -0.0680, -0.0918, -0.1060, -0.0893,
                        -0.1554, -0.0723, -0.0858, -0.0765, -0.0899, -0.0879, -0.0774, -0.0723,
                        -0.0917, -0.0638, -0.1018, -0.0856, -0.0885, -0.0788, -0.0887, -0.0931,
                        -0.0730, -0.1804, -0.0761, -0.1075, -0.0798, -0.0825, -0.0737, -0.1257,
                        -0.1027, -0.0811, -0.1061, -0.1098, -0.0930, -0.0941, -0.0864, -0.2104,
                        -0.0827, -0.1401, -0.0697, -0.1268, -0.0916, -0.0902, -0.1025, -0.0864,
                        -0.1563, -0.0974, -0.1392, -0.0760, -0.1081, -0.0791, -0.1267, -0.0792,
                        -0.0846, -0.0974, -0.1029, -0.1002, -0.0795, -0.0805, -0.1006, -0.0912,
                        -0.0909, -0.0923, -0.1016, -0.0768, -0.0939, -0.0717, -0.0819, -0.0890,
                        -0.0892, -0.0794, -0.0970, -0.0806, -0.0850, -0.0969, -0.0797, -0.1159,
                        -0.1136, -0.1209, -0.1756, -0.1098, -0.0980, -0.0732, -0.0825, -0.1262,
                        -0.0791, -0.0718, -0.0825, -0.0926, -0.0971, -0.1017, -0.0876, -0.1162,
                        -0.0832, -0.0837, -0.1025, -0.1287, -0.0916, -0.1175, -0.0851, -0.0762,
                        -0.1179, -0.0911, -0.0860, -0.1125, -0.1527, -0.0842, -0.0924, -0.1113,
                        -0.1188, -0.1113, -0.0728, -0.1016, -0.0886, -0.0867, -0.1397, -0.0926,
                        -0.0732, -0.0990, -0.0827, -0.0932, -0.0791, -0.0793, -0.0919, -0.1269,
                        -0.0961, -0.0834, -0.0985, -0.1180, -0.0996, -0.0848, -0.1071, -0.0941,
                        -0.0841, -0.0933, -0.0830, -0.0747, -0.0948, -0.0815, -0.1669, -0.0728,
                        -0.0872, -0.0737, -0.1027, -0.1341, -0.1239, -0.0997, -0.0938, -0.0785,
                        -0.1020, -0.0888, -0.0809, -0.0754, -0.1165, -0.1068, -0.1183, -0.1318,
                        -0.0752, -0.1011, -0.0887, -0.0816, -0.0867, -0.1533, -0.0682, -0.0826,
                        -0.1181, -0.0794, -0.1811, -0.0794, -0.0825, -0.1123, -0.0829, -0.0726,
                        -0.1259, -0.1708, -0.1451, -0.0970, -0.1754, -0.1387, -0.0939, -0.1013,
                        -0.0943, -0.0820, -0.0978, -0.0868, -0.1066, -0.1345, -0.1171, -0.1304,
                        -0.0801, -0.0860, -0.0800, -0.1053, -0.0781, -0.0908, -0.0963, -0.0954,
                        -0.1017, -0.1160, -0.1169, -0.0951, -0.0649, -0.0824, -0.1065, -0.0815,
                        -0.0991, -0.0815, -0.0800, -0.0878, -0.1374, -0.1262, -0.0971, -0.0818,
                        -0.1502, -0.0947, -0.0853, -0.0771, -0.0831, -0.0786, -0.1780, -0.0867,
                        -0.1150, -0.0842, -0.0867, -0.0894, -0.0974, -0.1025, -0.1101, -0.0836,
                        -0.1057, -0.0895, -0.0965, -0.0727, -0.1306, -0.0947, -0.0844, -0.0794,
                        -0.0980, -0.0895, -0.0846, -0.0747, -0.0940, -0.0781, -0.0847, -0.0911,
                        -0.0927, -0.1313, -0.0963, -0.0934, -0.1382, -0.0777, -0.0828, -0.0782,
                        -0.1006, -0.0684, -0.0952, -0.1494, -0.1025, -0.1161, -0.0873, -0.0773,
                        -0.1036, -0.0766, -0.1330, -0.1002, -0.1001]), max_val=tensor([0.0709, 0.0866, 0.0992, 0.0899, 0.0832, 0.0925, 0.1134, 0.0865, 0.0768,
                        0.0721, 0.0747, 0.0744, 0.0953, 0.0869, 0.0831, 0.0727, 0.0694, 0.0820,
                        0.0877, 0.0817, 0.0846, 0.0709, 0.0711, 0.0820, 0.0693, 0.0848, 0.0917,
                        0.0698, 0.0845, 0.0743, 0.0858, 0.0814, 0.0714, 0.1009, 0.0764, 0.0785,
                        0.0769, 0.1077, 0.0680, 0.0802, 0.0809, 0.0769, 0.0814, 0.0937, 0.0701,
                        0.0773, 0.0803, 0.0782, 0.1081, 0.0796, 0.0707, 0.0728, 0.0679, 0.0685,
                        0.0823, 0.0784, 0.0828, 0.0848, 0.0701, 0.0783, 0.0733, 0.0697, 0.0832,
                        0.0776, 0.0821, 0.1295, 0.0822, 0.0825, 0.0873, 0.1638, 0.0846, 0.0884,
                        0.1028, 0.0651, 0.1226, 0.0716, 0.0773, 0.0902, 0.0712, 0.0916, 0.1009,
                        0.0767, 0.0670, 0.0838, 0.0789, 0.0804, 0.0795, 0.0744, 0.0793, 0.0867,
                        0.0715, 0.1088, 0.0803, 0.0794, 0.0724, 0.0813, 0.0797, 0.0973, 0.0670,
                        0.0735, 0.0822, 0.0757, 0.0885, 0.2182, 0.0793, 0.0864, 0.0771, 0.0872,
                        0.0717, 0.0829, 0.0674, 0.0953, 0.0715, 0.0826, 0.0727, 0.1251, 0.1378,
                        0.0747, 0.0730, 0.0859, 0.0739, 0.0809, 0.0755, 0.0727, 0.0809, 0.0817,
                        0.0843, 0.0732, 0.0761, 0.0701, 0.1070, 0.0768, 0.0775, 0.0991, 0.0719,
                        0.0738, 0.0760, 0.0800, 0.0948, 0.0931, 0.0631, 0.0721, 0.0816, 0.0676,
                        0.0823, 0.1291, 0.0778, 0.0899, 0.1105, 0.0866, 0.0710, 0.0744, 0.0824,
                        0.0902, 0.0947, 0.0836, 0.0641, 0.1012, 0.0733, 0.0832, 0.0710, 0.0890,
                        0.0681, 0.0777, 0.0785, 0.0977, 0.0738, 0.0772, 0.0734, 0.0734, 0.0827,
                        0.0845, 0.0786, 0.0764, 0.0891, 0.0798, 0.1037, 0.0754, 0.0745, 0.0698,
                        0.0717, 0.1215, 0.0911, 0.0958, 0.0870, 0.0722, 0.0842, 0.0737, 0.0765,
                        0.0828, 0.0794, 0.0864, 0.0709, 0.0740, 0.0800, 0.0974, 0.0843, 0.0793,
                        0.0990, 0.0832, 0.0866, 0.0718, 0.0804, 0.1439, 0.1235, 0.0868, 0.0765,
                        0.0809, 0.0728, 0.0834, 0.0931, 0.0881, 0.1028, 0.0711, 0.0751, 0.0792,
                        0.0638, 0.0729, 0.1181, 0.0793, 0.0854, 0.0866, 0.0807, 0.1041, 0.0986,
                        0.0741, 0.0832, 0.0706, 0.0967, 0.0750, 0.1033, 0.0804, 0.0780, 0.0794,
                        0.0677, 0.0686, 0.0765, 0.0900, 0.0792, 0.0741, 0.0715, 0.0873, 0.0901,
                        0.1775, 0.0955, 0.0650, 0.0813, 0.0838, 0.1056, 0.0982, 0.0940, 0.0995,
                        0.0862, 0.0758, 0.0780, 0.0942, 0.0797, 0.0763, 0.0684, 0.0716, 0.0720,
                        0.0738, 0.0996, 0.0766, 0.0858, 0.1279, 0.0769, 0.1409, 0.0760, 0.0777,
                        0.0756, 0.0767, 0.1112, 0.0715, 0.0689, 0.0802, 0.0893, 0.0733, 0.0702,
                        0.0852, 0.0775, 0.1053, 0.0743, 0.1096, 0.1076, 0.0905, 0.0754, 0.0798,
                        0.0837, 0.0804, 0.1972, 0.0715, 0.0824, 0.0705, 0.0896, 0.0787, 0.0717,
                        0.1238, 0.0843, 0.0787, 0.0823, 0.1141, 0.0826, 0.0822, 0.0713, 0.0728,
                        0.2087, 0.0752, 0.0761, 0.0790, 0.0905, 0.1069, 0.0893, 0.0805, 0.0805,
                        0.1042, 0.0838, 0.0855, 0.0781, 0.0895, 0.0691, 0.0760, 0.1805, 0.0838,
                        0.0696, 0.0662, 0.0682, 0.0996, 0.0901, 0.0713, 0.2711, 0.0721, 0.0763,
                        0.0753, 0.0661, 0.0765, 0.0754, 0.0779, 0.0891, 0.0966, 0.0829, 0.0820,
                        0.0719, 0.0882, 0.0708, 0.0746, 0.0823, 0.0795, 0.1123, 0.0711, 0.0779,
                        0.0943, 0.1039, 0.1099, 0.0760, 0.0720, 0.0823, 0.0770, 0.0744, 0.0748,
                        0.0837, 0.0698, 0.1021, 0.0837, 0.0879, 0.0813, 0.0814, 0.0997, 0.0722,
                        0.0716, 0.0806, 0.0755, 0.0844, 0.0744, 0.0695, 0.1119, 0.0785, 0.0752,
                        0.0801, 0.0923, 0.0922, 0.0881, 0.1078, 0.0770, 0.1144, 0.0893, 0.0735,
                        0.0854, 0.0831, 0.0839, 0.0741, 0.0705, 0.0818, 0.0734, 0.0763, 0.0947,
                        0.0779, 0.0812, 0.0769, 0.0816, 0.0718, 0.0686, 0.0698, 0.0883, 0.0839,
                        0.1464, 0.0766, 0.0733, 0.0774, 0.0707, 0.1427, 0.0770, 0.0854, 0.0853,
                        0.0772, 0.0789, 0.0952, 0.0977, 0.0894, 0.0899, 0.0895, 0.0791, 0.0829,
                        0.0726, 0.0751, 0.0782, 0.0706, 0.0870, 0.0873, 0.0821, 0.0724, 0.0872,
                        0.0934, 0.0682, 0.0883, 0.0946, 0.0957, 0.0859, 0.0807, 0.0773, 0.1128,
                        0.0724, 0.0742, 0.0788, 0.0786, 0.0826, 0.0696, 0.0788, 0.0873, 0.1129,
                        0.0788, 0.0851, 0.0798, 0.0728, 0.0822, 0.0828, 0.1252, 0.0780, 0.0753,
                        0.0660, 0.0821, 0.0859, 0.1269, 0.0738, 0.0936, 0.1081, 0.0772, 0.0799,
                        0.0706, 0.1145, 0.0876, 0.0859, 0.0747, 0.0769, 0.0713, 0.1080, 0.0734,
                        0.0800, 0.0672, 0.0868, 0.0789, 0.0863, 0.0816, 0.0774, 0.0682, 0.0719,
                        0.0968, 0.0749, 0.0898, 0.0722, 0.0853, 0.0860, 0.0812, 0.0713, 0.0680,
                        0.0667, 0.0803, 0.0717, 0.0887, 0.0945, 0.0780, 0.0688, 0.0696, 0.0928,
                        0.1287, 0.0806, 0.0685, 0.0766, 0.0796])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.2874]), zero_point=tensor([47], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-13.37441635131836, max_val=23.12215232849121)
            )
          )
          (bn1): BatchNorm2d(
            509, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0776]), zero_point=tensor([66], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-5.097564697265625, max_val=4.762815952301025)
            )
          )
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(
            509, 503, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0006, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0009, 0.0009,
                      0.0008, 0.0008, 0.0007, 0.0008, 0.0012, 0.0008, 0.0007, 0.0009, 0.0007,
                      0.0011, 0.0010, 0.0009, 0.0008, 0.0011, 0.0010, 0.0007, 0.0016, 0.0006,
                      0.0008, 0.0011, 0.0009, 0.0008, 0.0011, 0.0011, 0.0007, 0.0012, 0.0015,
                      0.0009, 0.0012, 0.0007, 0.0009, 0.0007, 0.0006, 0.0010, 0.0009, 0.0008,
                      0.0009, 0.0007, 0.0008, 0.0012, 0.0007, 0.0008, 0.0007, 0.0007, 0.0018,
                      0.0008, 0.0009, 0.0007, 0.0006, 0.0013, 0.0015, 0.0008, 0.0010, 0.0007,
                      0.0007, 0.0008, 0.0009, 0.0011, 0.0010, 0.0008, 0.0007, 0.0012, 0.0006,
                      0.0007, 0.0007, 0.0006, 0.0011, 0.0008, 0.0010, 0.0015, 0.0009, 0.0008,
                      0.0008, 0.0009, 0.0008, 0.0015, 0.0010, 0.0013, 0.0010, 0.0010, 0.0008,
                      0.0010, 0.0007, 0.0007, 0.0013, 0.0007, 0.0009, 0.0007, 0.0009, 0.0009,
                      0.0010, 0.0009, 0.0010, 0.0010, 0.0007, 0.0007, 0.0007, 0.0010, 0.0006,
                      0.0010, 0.0013, 0.0007, 0.0012, 0.0016, 0.0006, 0.0009, 0.0009, 0.0007,
                      0.0006, 0.0008, 0.0008, 0.0006, 0.0008, 0.0006, 0.0011, 0.0008, 0.0008,
                      0.0008, 0.0010, 0.0009, 0.0012, 0.0008, 0.0011, 0.0009, 0.0007, 0.0010,
                      0.0008, 0.0008, 0.0007, 0.0008, 0.0008, 0.0010, 0.0015, 0.0009, 0.0008,
                      0.0013, 0.0007, 0.0010, 0.0007, 0.0011, 0.0008, 0.0010, 0.0011, 0.0009,
                      0.0009, 0.0010, 0.0009, 0.0014, 0.0010, 0.0014, 0.0007, 0.0009, 0.0008,
                      0.0013, 0.0008, 0.0010, 0.0016, 0.0007, 0.0012, 0.0010, 0.0008, 0.0008,
                      0.0009, 0.0007, 0.0007, 0.0008, 0.0012, 0.0010, 0.0013, 0.0017, 0.0008,
                      0.0008, 0.0008, 0.0014, 0.0012, 0.0011, 0.0006, 0.0007, 0.0008, 0.0009,
                      0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0009, 0.0009, 0.0007, 0.0009,
                      0.0007, 0.0014, 0.0007, 0.0012, 0.0009, 0.0007, 0.0008, 0.0011, 0.0008,
                      0.0007, 0.0007, 0.0008, 0.0013, 0.0007, 0.0008, 0.0009, 0.0007, 0.0011,
                      0.0009, 0.0008, 0.0007, 0.0009, 0.0007, 0.0011, 0.0008, 0.0010, 0.0006,
                      0.0007, 0.0010, 0.0014, 0.0007, 0.0009, 0.0007, 0.0013, 0.0012, 0.0012,
                      0.0013, 0.0012, 0.0010, 0.0007, 0.0008, 0.0008, 0.0009, 0.0016, 0.0008,
                      0.0008, 0.0008, 0.0008, 0.0008, 0.0009, 0.0015, 0.0007, 0.0014, 0.0007,
                      0.0009, 0.0009, 0.0010, 0.0010, 0.0010, 0.0011, 0.0008, 0.0009, 0.0010,
                      0.0011, 0.0007, 0.0013, 0.0007, 0.0012, 0.0006, 0.0011, 0.0008, 0.0009,
                      0.0007, 0.0008, 0.0007, 0.0010, 0.0012, 0.0009, 0.0013, 0.0015, 0.0011,
                      0.0011, 0.0011, 0.0015, 0.0010, 0.0008, 0.0008, 0.0012, 0.0009, 0.0009,
                      0.0009, 0.0007, 0.0009, 0.0015, 0.0017, 0.0009, 0.0008, 0.0011, 0.0008,
                      0.0015, 0.0018, 0.0008, 0.0007, 0.0006, 0.0008, 0.0008, 0.0007, 0.0008,
                      0.0009, 0.0011, 0.0012, 0.0007, 0.0018, 0.0014, 0.0011, 0.0009, 0.0008,
                      0.0007, 0.0013, 0.0008, 0.0009, 0.0010, 0.0007, 0.0009, 0.0009, 0.0006,
                      0.0012, 0.0016, 0.0010, 0.0008, 0.0010, 0.0007, 0.0009, 0.0006, 0.0007,
                      0.0007, 0.0008, 0.0008, 0.0013, 0.0008, 0.0007, 0.0011, 0.0012, 0.0011,
                      0.0014, 0.0016, 0.0011, 0.0008, 0.0007, 0.0011, 0.0008, 0.0011, 0.0013,
                      0.0014, 0.0008, 0.0011, 0.0008, 0.0008, 0.0010, 0.0011, 0.0011, 0.0010,
                      0.0006, 0.0008, 0.0010, 0.0007, 0.0010, 0.0008, 0.0007, 0.0010, 0.0006,
                      0.0007, 0.0008, 0.0013, 0.0014, 0.0007, 0.0009, 0.0009, 0.0008, 0.0011,
                      0.0007, 0.0007, 0.0012, 0.0011, 0.0007, 0.0010, 0.0007, 0.0009, 0.0010,
                      0.0013, 0.0009, 0.0016, 0.0010, 0.0008, 0.0007, 0.0008, 0.0006, 0.0007,
                      0.0011, 0.0012, 0.0009, 0.0009, 0.0012, 0.0010, 0.0007, 0.0008, 0.0007,
                      0.0013, 0.0007, 0.0009, 0.0008, 0.0012, 0.0007, 0.0009, 0.0007, 0.0008,
                      0.0007, 0.0007, 0.0009, 0.0009, 0.0008, 0.0007, 0.0012, 0.0007, 0.0007,
                      0.0007, 0.0008, 0.0013, 0.0009, 0.0010, 0.0009, 0.0008, 0.0010, 0.0009,
                      0.0008, 0.0007, 0.0014, 0.0009, 0.0007, 0.0013, 0.0012, 0.0010, 0.0009,
                      0.0009, 0.0012, 0.0006, 0.0006, 0.0008, 0.0009, 0.0012, 0.0014, 0.0007,
                      0.0012, 0.0009, 0.0010, 0.0013, 0.0008, 0.0006, 0.0007, 0.0007, 0.0009,
                      0.0009, 0.0010, 0.0014, 0.0008, 0.0007, 0.0009, 0.0011, 0.0007, 0.0010,
                      0.0007, 0.0007, 0.0009, 0.0009, 0.0007, 0.0013, 0.0019, 0.0009, 0.0011,
                      0.0007, 0.0008, 0.0007, 0.0006, 0.0008, 0.0010, 0.0009, 0.0010, 0.0007,
                      0.0009, 0.0007, 0.0010, 0.0007, 0.0006, 0.0010, 0.0011, 0.0007, 0.0007,
                      0.0012, 0.0009, 0.0010, 0.0007, 0.0007, 0.0008, 0.0011, 0.0006]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
              (activation_post_process): MovingAveragePerChannelMinMaxObserver(
                min_val=tensor([-0.0760, -0.0929, -0.1157, -0.0805, -0.1380, -0.0823, -0.0777, -0.0747,
                        -0.1125, -0.1055, -0.1084, -0.0853, -0.0971, -0.1494, -0.1050, -0.0736,
                        -0.1158, -0.0832, -0.0835, -0.0818, -0.1182, -0.0824, -0.1324, -0.1080,
                        -0.0883, -0.0868, -0.0768, -0.1026, -0.1284, -0.1179, -0.0989, -0.1428,
                        -0.1108, -0.0915, -0.1381, -0.1857, -0.0894, -0.1475, -0.0894, -0.1206,
                        -0.0953, -0.0826, -0.1267, -0.1159, -0.1055, -0.1170, -0.0863, -0.0852,
                        -0.1474, -0.0850, -0.0970, -0.0930, -0.0793, -0.0765, -0.0963, -0.1161,
                        -0.0839, -0.0805, -0.1665, -0.1913, -0.0986, -0.1279, -0.0905, -0.0812,
                        -0.0775, -0.1130, -0.1396, -0.1326, -0.1074, -0.0925, -0.1473, -0.0812,
                        -0.0950, -0.0912, -0.0829, -0.0942, -0.0981, -0.1234, -0.1963, -0.1205,
                        -0.1050, -0.0997, -0.1142, -0.1041, -0.1092, -0.1023, -0.1658, -0.0740,
                        -0.1320, -0.0961, -0.1281, -0.0789, -0.0830, -0.1175, -0.0791, -0.1090,
                        -0.0758, -0.1189, -0.0962, -0.1128, -0.1156, -0.0737, -0.1338, -0.0843,
                        -0.0843, -0.0821, -0.0924, -0.0769, -0.1242, -0.1603, -0.0671, -0.1501,
                        -0.2006, -0.0777, -0.1164, -0.0839, -0.0846, -0.0788, -0.1006, -0.1000,
                        -0.0794, -0.1001, -0.0830, -0.1455, -0.1040, -0.0684, -0.1068, -0.1163,
                        -0.1162, -0.0919, -0.0908, -0.1421, -0.0841, -0.0807, -0.1222, -0.0706,
                        -0.1015, -0.0947, -0.1050, -0.1061, -0.1257, -0.1939, -0.0754, -0.0859,
                        -0.1700, -0.0922, -0.1237, -0.0930, -0.0843, -0.0997, -0.1321, -0.1406,
                        -0.0741, -0.0824, -0.0871, -0.1215, -0.0781, -0.1284, -0.0818, -0.0743,
                        -0.1214, -0.0861, -0.1665, -0.0981, -0.1292, -0.1887, -0.0945, -0.1503,
                        -0.0799, -0.0997, -0.0968, -0.1120, -0.0915, -0.0851, -0.1010, -0.1585,
                        -0.0783, -0.1610, -0.2143, -0.1061, -0.0989, -0.1071, -0.0774, -0.1489,
                        -0.0848, -0.0772, -0.0928, -0.0995, -0.1214, -0.0998, -0.1123, -0.0922,
                        -0.1098, -0.1051, -0.0802, -0.0786, -0.0713, -0.1200, -0.0896, -0.1833,
                        -0.0849, -0.1473, -0.1136, -0.0933, -0.0876, -0.1345, -0.0786, -0.0745,
                        -0.0934, -0.0983, -0.1696, -0.0850, -0.0765, -0.1094, -0.0833, -0.1420,
                        -0.1200, -0.1024, -0.0959, -0.1109, -0.0843, -0.0941, -0.1068, -0.1325,
                        -0.0831, -0.0747, -0.1303, -0.1777, -0.0836, -0.0875, -0.0889, -0.1018,
                        -0.0740, -0.1578, -0.0847, -0.1479, -0.0973, -0.0942, -0.0992, -0.0900,
                        -0.1201, -0.2051, -0.1002, -0.0965, -0.1034, -0.0728, -0.0770, -0.0778,
                        -0.0975, -0.0847, -0.1847, -0.0807, -0.1159, -0.1134, -0.1304, -0.0786,
                        -0.1177, -0.0834, -0.0868, -0.0864, -0.0871, -0.0688, -0.0706, -0.0799,
                        -0.0848, -0.1597, -0.0768, -0.1453, -0.0751, -0.0736, -0.0863, -0.0872,
                        -0.0842, -0.1309, -0.1097, -0.0743, -0.1357, -0.0819, -0.1433, -0.0998,
                        -0.0868, -0.1168, -0.1283, -0.0836, -0.1075, -0.1559, -0.1213, -0.0880,
                        -0.0999, -0.0943, -0.1045, -0.1862, -0.2121, -0.0885, -0.0972, -0.1460,
                        -0.0986, -0.1877, -0.2327, -0.0916, -0.0926, -0.0829, -0.1068, -0.0936,
                        -0.0805, -0.1033, -0.1114, -0.0892, -0.1546, -0.0807, -0.2285, -0.1212,
                        -0.1381, -0.0820, -0.1028, -0.0859, -0.0976, -0.0956, -0.1132, -0.1281,
                        -0.0725, -0.1159, -0.1130, -0.0766, -0.1520, -0.1995, -0.1275, -0.1027,
                        -0.0941, -0.0838, -0.0691, -0.0821, -0.0837, -0.0746, -0.0905, -0.1017,
                        -0.1611, -0.0797, -0.0842, -0.1458, -0.1585, -0.1386, -0.1602, -0.1999,
                        -0.1253, -0.1031, -0.0866, -0.1349, -0.1019, -0.0898, -0.1060, -0.1094,
                        -0.0899, -0.0796, -0.0877, -0.0738, -0.0802, -0.0743, -0.0946, -0.0752,
                        -0.0809, -0.0887, -0.1044, -0.0945, -0.0962, -0.0785, -0.0801, -0.0759,
                        -0.0731, -0.0856, -0.1003, -0.0905, -0.1331, -0.0734, -0.0900, -0.0971,
                        -0.0788, -0.1385, -0.0849, -0.0812, -0.1594, -0.0787, -0.0903, -0.0845,
                        -0.0947, -0.0759, -0.1300, -0.1659, -0.1136, -0.1184, -0.1219, -0.1048,
                        -0.0890, -0.1057, -0.0811, -0.0820, -0.1438, -0.1507, -0.0748, -0.1211,
                        -0.1123, -0.1323, -0.0782, -0.0822, -0.0818, -0.1700, -0.0930, -0.0878,
                        -0.0882, -0.1492, -0.0815, -0.0783, -0.0916, -0.0731, -0.0845, -0.0897,
                        -0.1200, -0.1161, -0.0852, -0.0703, -0.1195, -0.0809, -0.0908, -0.0698,
                        -0.1012, -0.1662, -0.1115, -0.1339, -0.1111, -0.1080, -0.1128, -0.0886,
                        -0.0677, -0.0791, -0.1754, -0.1146, -0.0843, -0.1166, -0.1520, -0.1068,
                        -0.1148, -0.1153, -0.1504, -0.0759, -0.0747, -0.1013, -0.1118, -0.1442,
                        -0.1754, -0.0853, -0.0926, -0.1184, -0.1260, -0.0821, -0.1078, -0.0825,
                        -0.0701, -0.0687, -0.1188, -0.1158, -0.1198, -0.1256, -0.1061, -0.0763,
                        -0.0938, -0.0874, -0.0948, -0.0848, -0.0846, -0.0900, -0.1126, -0.1049,
                        -0.0871, -0.1713, -0.1258, -0.0768, -0.0855, -0.0819, -0.0849, -0.0954,
                        -0.0800, -0.0950, -0.1296, -0.1110, -0.0770, -0.0879, -0.1112, -0.0959,
                        -0.1337, -0.0957, -0.0734, -0.1308, -0.1379, -0.0756, -0.0825, -0.1562,
                        -0.0883, -0.1234, -0.0789, -0.0704, -0.0703, -0.0801, -0.0723]), max_val=tensor([0.0816, 0.1486, 0.1372, 0.1449, 0.0704, 0.1390, 0.1339, 0.1108, 0.0987,
                        0.0797, 0.0914, 0.0882, 0.0859, 0.0792, 0.0838, 0.0863, 0.1180, 0.0727,
                        0.1363, 0.1265, 0.0948, 0.1055, 0.1345, 0.1273, 0.0871, 0.2085, 0.0727,
                        0.0793, 0.1358, 0.1024, 0.0817, 0.0846, 0.1442, 0.0752, 0.1570, 0.0833,
                        0.1127, 0.0928, 0.0818, 0.0750, 0.0785, 0.0758, 0.0845, 0.1197, 0.0795,
                        0.1145, 0.0737, 0.0955, 0.0948, 0.0830, 0.0951, 0.0867, 0.0831, 0.2336,
                        0.0846, 0.1198, 0.0828, 0.0725, 0.0751, 0.0823, 0.0902, 0.0898, 0.0727,
                        0.0895, 0.1021, 0.0786, 0.0769, 0.0677, 0.0873, 0.0825, 0.0890, 0.0735,
                        0.0896, 0.0729, 0.0748, 0.1400, 0.0920, 0.1062, 0.1013, 0.1149, 0.1008,
                        0.0949, 0.1154, 0.0819, 0.1930, 0.1213, 0.0899, 0.1254, 0.1186, 0.0764,
                        0.0815, 0.0827, 0.0864, 0.1659, 0.0939, 0.0826, 0.0923, 0.1052, 0.1085,
                        0.1328, 0.1077, 0.1316, 0.1130, 0.0817, 0.0794, 0.0926, 0.1287, 0.0790,
                        0.1290, 0.1594, 0.0950, 0.1233, 0.1706, 0.0660, 0.0873, 0.1105, 0.0774,
                        0.0792, 0.0724, 0.0877, 0.0797, 0.0910, 0.0723, 0.0849, 0.0893, 0.1025,
                        0.0649, 0.1220, 0.0679, 0.1540, 0.1002, 0.1279, 0.1090, 0.0946, 0.1002,
                        0.1074, 0.0855, 0.0839, 0.0711, 0.0929, 0.0779, 0.1114, 0.1150, 0.1009,
                        0.1026, 0.0870, 0.1150, 0.0877, 0.1395, 0.0897, 0.0995, 0.0833, 0.1127,
                        0.1116, 0.1209, 0.0798, 0.1821, 0.0824, 0.1756, 0.0864, 0.0812, 0.0995,
                        0.0775, 0.0980, 0.0843, 0.2022, 0.0733, 0.1206, 0.1274, 0.0730, 0.0724,
                        0.0776, 0.0762, 0.0898, 0.0824, 0.1452, 0.1310, 0.1565, 0.0719, 0.0784,
                        0.0991, 0.0749, 0.1747, 0.0825, 0.1456, 0.0747, 0.0713, 0.0750, 0.1074,
                        0.1142, 0.0870, 0.1146, 0.0962, 0.0981, 0.1118, 0.1111, 0.0927, 0.0761,
                        0.0823, 0.0933, 0.0850, 0.1321, 0.0999, 0.0704, 0.1033, 0.1423, 0.0955,
                        0.0826, 0.0903, 0.1027, 0.1150, 0.0843, 0.1053, 0.1148, 0.0764, 0.1232,
                        0.1033, 0.0926, 0.0851, 0.0932, 0.0769, 0.1350, 0.0856, 0.0849, 0.0819,
                        0.0843, 0.0803, 0.0896, 0.0815, 0.1168, 0.0878, 0.1634, 0.1519, 0.0857,
                        0.1602, 0.0762, 0.1213, 0.0825, 0.0943, 0.1021, 0.1072, 0.0988, 0.0781,
                        0.0931, 0.0761, 0.1054, 0.1005, 0.1131, 0.1863, 0.0798, 0.1721, 0.0858,
                        0.0832, 0.1113, 0.0781, 0.1310, 0.1255, 0.1375, 0.0961, 0.1131, 0.1255,
                        0.1373, 0.0921, 0.1590, 0.0814, 0.1368, 0.0777, 0.0807, 0.1013, 0.1081,
                        0.0873, 0.1049, 0.0743, 0.0838, 0.1521, 0.1192, 0.1669, 0.1863, 0.0940,
                        0.1384, 0.1447, 0.1868, 0.0887, 0.0995, 0.0875, 0.1075, 0.0992, 0.1085,
                        0.1086, 0.0719, 0.1101, 0.0971, 0.1001, 0.1092, 0.0853, 0.0812, 0.0944,
                        0.1123, 0.0704, 0.1079, 0.0830, 0.0709, 0.0676, 0.1073, 0.0846, 0.0707,
                        0.0973, 0.1460, 0.1403, 0.0939, 0.1745, 0.1767, 0.0927, 0.1172, 0.0770,
                        0.0803, 0.1594, 0.1077, 0.0898, 0.1154, 0.0882, 0.0949, 0.0748, 0.0812,
                        0.0753, 0.0859, 0.0715, 0.0754, 0.1300, 0.0832, 0.1096, 0.0788, 0.0807,
                        0.0834, 0.1017, 0.1078, 0.1122, 0.0969, 0.0759, 0.1210, 0.0835, 0.1006,
                        0.1763, 0.1122, 0.1381, 0.0852, 0.0886, 0.1150, 0.0946, 0.1340, 0.1701,
                        0.1828, 0.1066, 0.1365, 0.1036, 0.1040, 0.1234, 0.1362, 0.1447, 0.1259,
                        0.0793, 0.0995, 0.1272, 0.0767, 0.1212, 0.1028, 0.0894, 0.1305, 0.0814,
                        0.0851, 0.0762, 0.1690, 0.1803, 0.0939, 0.1090, 0.1082, 0.0962, 0.0873,
                        0.0777, 0.0902, 0.0781, 0.1398, 0.0749, 0.1307, 0.0702, 0.1125, 0.1216,
                        0.0802, 0.0849, 0.2021, 0.0998, 0.0802, 0.0822, 0.0773, 0.0765, 0.0949,
                        0.0683, 0.0924, 0.1128, 0.0918, 0.1530, 0.1312, 0.0887, 0.1061, 0.0845,
                        0.1246, 0.0774, 0.1165, 0.1042, 0.1563, 0.0900, 0.1089, 0.0900, 0.0967,
                        0.0913, 0.0825, 0.0687, 0.0872, 0.1035, 0.0873, 0.1544, 0.0831, 0.0761,
                        0.0942, 0.0832, 0.0954, 0.0881, 0.0849, 0.0811, 0.0971, 0.1323, 0.1156,
                        0.1026, 0.0914, 0.0758, 0.0951, 0.0756, 0.1708, 0.0746, 0.1255, 0.0814,
                        0.0779, 0.0926, 0.0802, 0.0784, 0.1022, 0.1203, 0.1568, 0.0996, 0.0764,
                        0.1524, 0.0895, 0.1256, 0.1659, 0.0772, 0.0772, 0.0846, 0.0846, 0.0778,
                        0.0802, 0.1306, 0.1756, 0.1008, 0.0891, 0.1098, 0.1371, 0.0769, 0.1219,
                        0.0769, 0.0929, 0.0783, 0.1126, 0.0873, 0.1496, 0.2354, 0.1096, 0.1362,
                        0.0836, 0.0993, 0.0848, 0.0719, 0.1032, 0.1308, 0.0804, 0.1212, 0.0886,
                        0.0929, 0.0800, 0.0797, 0.0707, 0.0783, 0.1055, 0.0931, 0.0910, 0.0838,
                        0.0790, 0.1094, 0.0806, 0.0836, 0.0828, 0.0989, 0.1425, 0.0742])
              )
            )
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.2181]), zero_point=tensor([70], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-15.206204414367676, max_val=12.489242553710938)
            )
          )
          (bn2): BatchNorm2d(
            503, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            (activation_post_process): FusedMovingAvgObsFakeQuantize(
              fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0064]), zero_point=tensor([94], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
              (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.6024205684661865, max_val=0.2111302614212036)
            )
          )
          (act2): ReLU(inplace=True)
        )
      )
      (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
      (fc): Linear(
        in_features=503, out_features=10, bias=True
        (weight_fake_quant): FusedMovingAvgObsFakeQuantize(
          fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0016, 0.0016, 0.0014, 0.0011, 0.0012, 0.0015, 0.0015, 0.0016, 0.0027,
                  0.0020]), zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32), dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, reduce_range=False
          (activation_post_process): MovingAveragePerChannelMinMaxObserver(
            min_val=tensor([-0.2064, -0.2094, -0.1804, -0.1433, -0.1559, -0.1977, -0.1981, -0.2053,
                    -0.3422, -0.2620]), max_val=tensor([0.1206, 0.1042, 0.0950, 0.0997, 0.0860, 0.0920, 0.0929, 0.0944, 0.1302,
                    0.0827])
          )
        )
        (activation_post_process): FusedMovingAvgObsFakeQuantize(
          fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.1499]), zero_point=tensor([67], dtype=torch.int32), dtype=torch.quint8, quant_min=0, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=True
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-10.058563232421875, max_val=8.980971336364746)
        )
      )
    )



.. code:: ipython3

    pruned_model_int8 = torch.quantization.convert(pruned_model_prepared)

.. code:: ipython3

    torch.save(pruned_model_int8.state_dict(), './weights/pruned_model_int8-weights.pth')
    torch.save(pruned_model_int8, './weights/pruned_model_int8.pth')

.. code:: ipython3

    
    # run the model, relevant calculations will happen in int8
    res = model_int8(input_fp32)
            
