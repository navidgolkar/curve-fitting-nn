# curve-fitting-nn
4 different configurations of neural networks used for curve fitting + visualization

in bash using arguments you can change:
#### 1. Structure Parameters:
___
-  --hn: Number of hidden layers [type=int, default=5]
-  --nn: number of nodes at each hidden layer (the number of layers is equal at all hidden layers) [type=int, default=7]
-  --func: 1-26: Which activation function to use, default is Mish [type=int, default=17] (to see which number corresponds to what activation function check paramters.py)
-  --conv: (kernel size, padding, stride) for convolutional neural network [type=tuple[int, int, int], default=(3, 1, 1)]
-  --connect: Number of connections for ConvResNet residual connections (connections start from layer+2) [type=int, default=1]
#### 2. Training Hyper-parameters:
___
-  --loss1: 1-9: Which loss unction to use for the training of models, default is Mean Squared Loss [type=int, default=2] (to see which number corresponds to what loss function, check paramters.py)
-  --opt: 1-12: Which optimizer to use for training, default is Adam [type=int, default=1] (to see which number corresponds to what optimizer function, check paramters.py)
-  --lr: Enter learning rate value [type=float, default=1e-2]
-  --grad_clip: Enter the value at which the gradient should be clipped to prevent explosion [type=float, default=100]
-  --tol: Enter the tolerance for the network at which to stop training [type=float, default=1e-3]
-  --epoch: Number of epochs to run [type=int, default=1000]
-  --shuffle: Will shuffle input data of models [action='store_true']
-  --device: What device to use for pytorch [type=str, default='cpu']
#### 3. Outputs and Plots Parameters:
___
-  --loss2: 1-9: Which loss unction to use for the second plot (this is not used for training), default is Binary Cross Entropy with Logits Loss [type=int, default=6] (to see which number corresponds to what loss function, check paramters.py)
-  --log: the results should be shown per how many epochs [type=int, default=10]
-  --verbose: Whether to show results in console [action='store_true']
-  --show: Whether to open figure files after running the code [action='store_true']
-  --file_type: What should be the file_type of saved figures (gif, png, jpeg) [type=str, default='gif']
-  --name: added string at the end of each file for keeping track at running multiple runs [type=str, default='']
#### 4. Input Data Parameters:
___
-  --in_n: Number of input data [type=int, default=200]
-  --in_std: Standard deviation for input noise [type=float, default=1e-1]
#### 5. Seed variable influences both training parameters and input data parameters
___
-  --seed: Seed number for random values [type=int, default=1]

# Plots
- Top left: data vs prediction
- Top Right: the graph of network (red connections mean negative weights, blue means positive, and opacity is based on the absolute value of the weights, so smaller weight, lower opacity)
- Bottom Left: Mean Squared Loss through epochs
- Bottom Right: Binary Cross Entropy Loss through epochs (this loss is calculated with the real y values that don't have noise)

to use you can clone this repository and install the packages needed in requirements.txt and run main.py
#### The input data formula: $2e^{-x}(\sin(5x)+x\cos(5x))$
| Dense Configuration | Convolutional Configuration |
| :-------: | :-------: |
| ![densenet](/saves/Mish/FCNN_7x7_1.gif) | ![convnet](/saves/Mish/CNN_7x7_1.gif) |
| ![denseresnet](/saves/Mish/DenseResNet_7x7_1.gif) | ![convresnet](/saves/Mish/ConvResNet_7x7_1.gif) |

| Custom Configuration |
| :------------------: |
| ![customnet](/saves/Mish/CustomNet_7x7_1.gif) |

## Observations
- FCNN and DenseResNet change more uniformly through epochs because of their dense configuration, while CNN, ConvResNet and CustomNet change very "noisily"
- These activation functions works best for all configurations: SiLU, Mish, GELU, SELU, Hardswish, Tanhshrink.
- MSELoss(), L1Loss(), HuberLoss(), SmoothL1Loss(), which have similar formulas, work best for this problem
- CustomNet, FCNN, and DenseResNet, albeit its dense configuration, converges best throughout all different hyperparameters

## To-do
- change the skip connections so that they're defined node by node, so that in pruning they can be pruned independently if needed
- ![Structure Optimization](https://github.com/navidgolkar/curve-fitting-nnwithopt): add a heuristic approach for optimizing a 
Neural Network (CustomNet) by pruning connections e.g.: Grey Wolves Optimization, Genetic Algorithm, etc.
- Optimizing CustomNet by checking different activation functions for each layer

## Console Output
```
------------------------------------------------------------
  FCNN_7x7_1
FCNN(
  (net): Sequential(
    (0): Linear(in_features=1, out_features=7, bias=True)
    (1): Mish()
    (2): Linear(in_features=7, out_features=7, bias=True)
    (3): Mish()
    (4): Linear(in_features=7, out_features=7, bias=True)
    (5): Mish()
    (6): Linear(in_features=7, out_features=7, bias=True)
    (7): Mish()
    (8): Linear(in_features=7, out_features=7, bias=True)
    (9): Mish()
  )
  (head): Linear(in_features=7, out_features=1, bias=True)
)
------------------------------------------------------------

Epoch      1 | MSELoss: 0.422843 | BCEWithLogitsLoss: 0.6368
Epoch     10 | MSELoss: 0.367435 | BCEWithLogitsLoss: 0.7472
Epoch     20 | MSELoss: 0.356272 | BCEWithLogitsLoss: 0.7060
Epoch     30 | MSELoss: 0.339159 | BCEWithLogitsLoss: 0.7082
Epoch     40 | MSELoss: 0.283609 | BCEWithLogitsLoss: 0.6735
Epoch     50 | MSELoss: 0.226037 | BCEWithLogitsLoss: 0.6037
Epoch     60 | MSELoss: 0.165765 | BCEWithLogitsLoss: 0.5587
Epoch     70 | MSELoss: 0.154065 | BCEWithLogitsLoss: 0.5181
Epoch     80 | MSELoss: 0.144663 | BCEWithLogitsLoss: 0.5518
Epoch     90 | MSELoss: 0.136738 | BCEWithLogitsLoss: 0.5501
Epoch    100 | MSELoss: 0.125114 | BCEWithLogitsLoss: 0.5313
Epoch    110 | MSELoss: 0.117332 | BCEWithLogitsLoss: 0.5282
Epoch    120 | MSELoss: 0.110708 | BCEWithLogitsLoss: 0.5039
Epoch    130 | MSELoss: 0.103779 | BCEWithLogitsLoss: 0.5326
Epoch    140 | MSELoss: 0.087473 | BCEWithLogitsLoss: 0.4981
Epoch    150 | MSELoss: 0.070072 | BCEWithLogitsLoss: 0.4788
Epoch    160 | MSELoss: 0.054333 | BCEWithLogitsLoss: 0.4720
Epoch    170 | MSELoss: 0.040922 | BCEWithLogitsLoss: 0.4496
Epoch    180 | MSELoss: 0.036556 | BCEWithLogitsLoss: 0.4579
Epoch    190 | MSELoss: 0.034195 | BCEWithLogitsLoss: 0.4412
Epoch    200 | MSELoss: 0.034133 | BCEWithLogitsLoss: 0.4539
Epoch    210 | MSELoss: 0.032615 | BCEWithLogitsLoss: 0.4444
Epoch    220 | MSELoss: 0.031908 | BCEWithLogitsLoss: 0.4359
Epoch    230 | MSELoss: 0.032000 | BCEWithLogitsLoss: 0.4507
Epoch    240 | MSELoss: 0.031039 | BCEWithLogitsLoss: 0.4464
Epoch    250 | MSELoss: 0.031974 | BCEWithLogitsLoss: 0.4255
Epoch    260 | MSELoss: 0.031655 | BCEWithLogitsLoss: 0.4550
Epoch    270 | MSELoss: 0.029861 | BCEWithLogitsLoss: 0.4360
Epoch    280 | MSELoss: 0.029482 | BCEWithLogitsLoss: 0.4357
Epoch    290 | MSELoss: 0.029367 | BCEWithLogitsLoss: 0.4320
Epoch    300 | MSELoss: 0.030615 | BCEWithLogitsLoss: 0.4259
Epoch    310 | MSELoss: 0.030902 | BCEWithLogitsLoss: 0.4577
Epoch    320 | MSELoss: 0.029320 | BCEWithLogitsLoss: 0.4503
Epoch    330 | MSELoss: 0.028605 | BCEWithLogitsLoss: 0.4456
Epoch    340 | MSELoss: 0.028003 | BCEWithLogitsLoss: 0.4416
Epoch    350 | MSELoss: 0.027618 | BCEWithLogitsLoss: 0.4339
Epoch    360 | MSELoss: 0.028857 | BCEWithLogitsLoss: 0.4247
Epoch    370 | MSELoss: 0.027368 | BCEWithLogitsLoss: 0.4330
Epoch    380 | MSELoss: 0.026240 | BCEWithLogitsLoss: 0.4356
Epoch    390 | MSELoss: 0.026212 | BCEWithLogitsLoss: 0.4268
Epoch    400 | MSELoss: 0.025190 | BCEWithLogitsLoss: 0.4289
Epoch    410 | MSELoss: 0.039857 | BCEWithLogitsLoss: 0.4947
Epoch    420 | MSELoss: 0.025657 | BCEWithLogitsLoss: 0.4286
Epoch    430 | MSELoss: 0.024371 | BCEWithLogitsLoss: 0.4222
Epoch    440 | MSELoss: 0.021363 | BCEWithLogitsLoss: 0.4368
Epoch    450 | MSELoss: 0.019518 | BCEWithLogitsLoss: 0.4409
Epoch    460 | MSELoss: 0.020115 | BCEWithLogitsLoss: 0.4348
Epoch    470 | MSELoss: 0.015230 | BCEWithLogitsLoss: 0.4312
Epoch    480 | MSELoss: 0.014054 | BCEWithLogitsLoss: 0.4448
Epoch    490 | MSELoss: 0.011579 | BCEWithLogitsLoss: 0.4353
Epoch    500 | MSELoss: 0.010632 | BCEWithLogitsLoss: 0.4360
Epoch    510 | MSELoss: 0.009827 | BCEWithLogitsLoss: 0.4301
Epoch    520 | MSELoss: 0.016906 | BCEWithLogitsLoss: 0.4098
Epoch    530 | MSELoss: 0.013520 | BCEWithLogitsLoss: 0.4273
Epoch    540 | MSELoss: 0.009649 | BCEWithLogitsLoss: 0.4139
Epoch    550 | MSELoss: 0.009005 | BCEWithLogitsLoss: 0.4187
Epoch    560 | MSELoss: 0.008880 | BCEWithLogitsLoss: 0.4194
Epoch    570 | MSELoss: 0.008802 | BCEWithLogitsLoss: 0.4192
Epoch    580 | MSELoss: 0.008751 | BCEWithLogitsLoss: 0.4183
Epoch    590 | MSELoss: 0.008690 | BCEWithLogitsLoss: 0.4192
Epoch    600 | MSELoss: 0.008651 | BCEWithLogitsLoss: 0.4196
Epoch    610 | MSELoss: 0.008616 | BCEWithLogitsLoss: 0.4196
Epoch    620 | MSELoss: 0.008587 | BCEWithLogitsLoss: 0.4193
Epoch    630 | MSELoss: 0.008562 | BCEWithLogitsLoss: 0.4191
Epoch    640 | MSELoss: 0.008540 | BCEWithLogitsLoss: 0.4191
Epoch    650 | MSELoss: 0.008521 | BCEWithLogitsLoss: 0.4191
Epoch    660 | MSELoss: 0.008504 | BCEWithLogitsLoss: 0.4191
Epoch    670 | MSELoss: 0.008488 | BCEWithLogitsLoss: 0.4190
Epoch    680 | MSELoss: 0.008492 | BCEWithLogitsLoss: 0.4189
Epoch    690 | MSELoss: 0.025465 | BCEWithLogitsLoss: 0.4180
Epoch    700 | MSELoss: 0.010663 | BCEWithLogitsLoss: 0.4401
Epoch    710 | MSELoss: 0.009085 | BCEWithLogitsLoss: 0.4201
Epoch    720 | MSELoss: 0.008555 | BCEWithLogitsLoss: 0.4203
Epoch    730 | MSELoss: 0.008534 | BCEWithLogitsLoss: 0.4161
Epoch    740 | MSELoss: 0.008480 | BCEWithLogitsLoss: 0.4184
Epoch    750 | MSELoss: 0.008459 | BCEWithLogitsLoss: 0.4194
Epoch    760 | MSELoss: 0.008432 | BCEWithLogitsLoss: 0.4192
Epoch    770 | MSELoss: 0.008417 | BCEWithLogitsLoss: 0.4191
Epoch    780 | MSELoss: 0.008409 | BCEWithLogitsLoss: 0.4186
Epoch    790 | MSELoss: 0.008400 | BCEWithLogitsLoss: 0.4191
Epoch    800 | MSELoss: 0.008394 | BCEWithLogitsLoss: 0.4194
Epoch    810 | MSELoss: 0.008416 | BCEWithLogitsLoss: 0.4212
Epoch    820 | MSELoss: 0.011237 | BCEWithLogitsLoss: 0.4413
Epoch    830 | MSELoss: 0.010794 | BCEWithLogitsLoss: 0.3997
Epoch    840 | MSELoss: 0.009085 | BCEWithLogitsLoss: 0.4087
Epoch    850 | MSELoss: 0.008631 | BCEWithLogitsLoss: 0.4122
Epoch    860 | MSELoss: 0.008452 | BCEWithLogitsLoss: 0.4148
Epoch    870 | MSELoss: 0.008375 | BCEWithLogitsLoss: 0.4173
Epoch    880 | MSELoss: 0.008352 | BCEWithLogitsLoss: 0.4190
Epoch    890 | MSELoss: 0.008351 | BCEWithLogitsLoss: 0.4197
Epoch    900 | MSELoss: 0.008343 | BCEWithLogitsLoss: 0.4191
Epoch    910 | MSELoss: 0.008339 | BCEWithLogitsLoss: 0.4185
Epoch    920 | MSELoss: 0.008335 | BCEWithLogitsLoss: 0.4191
Epoch    930 | MSELoss: 0.008428 | BCEWithLogitsLoss: 0.4200
Epoch    940 | MSELoss: 0.014914 | BCEWithLogitsLoss: 0.4293
Epoch    950 | MSELoss: 0.009873 | BCEWithLogitsLoss: 0.4269
Epoch    960 | MSELoss: 0.008559 | BCEWithLogitsLoss: 0.4215
Epoch    970 | MSELoss: 0.008347 | BCEWithLogitsLoss: 0.4182
Epoch    980 | MSELoss: 0.008375 | BCEWithLogitsLoss: 0.4181
Epoch    990 | MSELoss: 0.008349 | BCEWithLogitsLoss: 0.4190
Epoch   1000 | MSELoss: 0.008566 | BCEWithLogitsLoss: 0.4249
Epoch   1010 | MSELoss: 0.013794 | BCEWithLogitsLoss: 0.4509
Epoch   1020 | MSELoss: 0.009403 | BCEWithLogitsLoss: 0.4330
Epoch   1030 | MSELoss: 0.008755 | BCEWithLogitsLoss: 0.4276
Epoch   1040 | MSELoss: 0.008497 | BCEWithLogitsLoss: 0.4246
Epoch   1050 | MSELoss: 0.008374 | BCEWithLogitsLoss: 0.4223
Epoch   1060 | MSELoss: 0.008320 | BCEWithLogitsLoss: 0.4208
Epoch   1070 | MSELoss: 0.008297 | BCEWithLogitsLoss: 0.4195
Epoch   1080 | MSELoss: 0.008292 | BCEWithLogitsLoss: 0.4185
Epoch   1090 | MSELoss: 0.008289 | BCEWithLogitsLoss: 0.4183
Epoch   1100 | MSELoss: 0.008284 | BCEWithLogitsLoss: 0.4189
Epoch   1110 | MSELoss: 0.008280 | BCEWithLogitsLoss: 0.4190
Epoch   1120 | MSELoss: 0.008276 | BCEWithLogitsLoss: 0.4187
Epoch   1130 | MSELoss: 0.008274 | BCEWithLogitsLoss: 0.4191
Epoch   1140 | MSELoss: 0.008627 | BCEWithLogitsLoss: 0.4219
Epoch   1150 | MSELoss: 0.008392 | BCEWithLogitsLoss: 0.4230
Epoch   1160 | MSELoss: 0.009153 | BCEWithLogitsLoss: 0.4124
Epoch   1170 | MSELoss: 0.008391 | BCEWithLogitsLoss: 0.4189
Epoch   1180 | MSELoss: 0.008458 | BCEWithLogitsLoss: 0.4210
Epoch   1190 | MSELoss: 0.008277 | BCEWithLogitsLoss: 0.4182
Epoch   1200 | MSELoss: 0.008292 | BCEWithLogitsLoss: 0.4175
Epoch   1210 | MSELoss: 0.008313 | BCEWithLogitsLoss: 0.4159
Epoch   1220 | MSELoss: 0.012867 | BCEWithLogitsLoss: 0.3922
Epoch   1230 | MSELoss: 0.010732 | BCEWithLogitsLoss: 0.4401
Epoch   1240 | MSELoss: 0.009060 | BCEWithLogitsLoss: 0.4313
Epoch   1250 | MSELoss: 0.008387 | BCEWithLogitsLoss: 0.4237
Epoch   1260 | MSELoss: 0.008261 | BCEWithLogitsLoss: 0.4195
Epoch   1270 | MSELoss: 0.008252 | BCEWithLogitsLoss: 0.4185
Epoch   1280 | MSELoss: 0.008247 | BCEWithLogitsLoss: 0.4187
Epoch   1290 | MSELoss: 0.008243 | BCEWithLogitsLoss: 0.4191
Epoch   1300 | MSELoss: 0.008240 | BCEWithLogitsLoss: 0.4193
Epoch   1310 | MSELoss: 0.008235 | BCEWithLogitsLoss: 0.4192
Epoch   1320 | MSELoss: 0.008231 | BCEWithLogitsLoss: 0.4187
Epoch   1330 | MSELoss: 0.008227 | BCEWithLogitsLoss: 0.4187
Epoch   1340 | MSELoss: 0.008223 | BCEWithLogitsLoss: 0.4189
Epoch   1350 | MSELoss: 0.008219 | BCEWithLogitsLoss: 0.4188
Epoch   1360 | MSELoss: 0.008216 | BCEWithLogitsLoss: 0.4188
Epoch   1370 | MSELoss: 0.008212 | BCEWithLogitsLoss: 0.4188
Epoch   1380 | MSELoss: 0.008208 | BCEWithLogitsLoss: 0.4188
Epoch   1390 | MSELoss: 0.008204 | BCEWithLogitsLoss: 0.4189
Epoch   1400 | MSELoss: 0.008265 | BCEWithLogitsLoss: 0.4209
Epoch   1410 | MSELoss: 0.020960 | BCEWithLogitsLoss: 0.4553
Epoch   1420 | MSELoss: 0.009329 | BCEWithLogitsLoss: 0.4111
Epoch   1430 | MSELoss: 0.008544 | BCEWithLogitsLoss: 0.4192
Epoch   1440 | MSELoss: 0.008457 | BCEWithLogitsLoss: 0.4250
Epoch   1450 | MSELoss: 0.008300 | BCEWithLogitsLoss: 0.4184
Epoch   1460 | MSELoss: 0.008265 | BCEWithLogitsLoss: 0.4183
Epoch   1470 | MSELoss: 0.008227 | BCEWithLogitsLoss: 0.4181
Epoch   1480 | MSELoss: 0.008210 | BCEWithLogitsLoss: 0.4195
Epoch   1490 | MSELoss: 0.008201 | BCEWithLogitsLoss: 0.4187
Epoch   1500 | MSELoss: 0.008197 | BCEWithLogitsLoss: 0.4185

------------------------------------------------------------
  CNN_7x7_1
CNN(
  (conv): Sequential(
    (0): Conv1d(1, 7, kernel_size=(1,), stride=(1,))
    (1): Mish()
    (2): Conv1d(7, 7, kernel_size=(3,), stride=(1,), padding=(1,))
    (3): Mish()
    (4): Conv1d(7, 7, kernel_size=(3,), stride=(1,), padding=(1,))
    (5): Mish()
    (6): Conv1d(7, 7, kernel_size=(3,), stride=(1,), padding=(1,))
    (7): Mish()
    (8): Conv1d(7, 7, kernel_size=(3,), stride=(1,), padding=(1,))
    (9): Mish()
  )
  (head): Conv1d(7, 1, kernel_size=(1,), stride=(1,))
)
------------------------------------------------------------

Epoch      1 | MSELoss: 0.416947 | BCEWithLogitsLoss: 0.6393
Epoch     10 | MSELoss: 0.363240 | BCEWithLogitsLoss: 0.7293
Epoch     20 | MSELoss: 0.354423 | BCEWithLogitsLoss: 0.7198
Epoch     30 | MSELoss: 0.338986 | BCEWithLogitsLoss: 0.6991
Epoch     40 | MSELoss: 0.302619 | BCEWithLogitsLoss: 0.6740
Epoch     50 | MSELoss: 0.268367 | BCEWithLogitsLoss: 0.6526
Epoch     60 | MSELoss: 0.247142 | BCEWithLogitsLoss: 0.6340
Epoch     70 | MSELoss: 0.234592 | BCEWithLogitsLoss: 0.6219
Epoch     80 | MSELoss: 0.223002 | BCEWithLogitsLoss: 0.6140
Epoch     90 | MSELoss: 0.201354 | BCEWithLogitsLoss: 0.6015
Epoch    100 | MSELoss: 0.161064 | BCEWithLogitsLoss: 0.5832
Epoch    110 | MSELoss: 0.151948 | BCEWithLogitsLoss: 0.5440
Epoch    120 | MSELoss: 0.138407 | BCEWithLogitsLoss: 0.5174
Epoch    130 | MSELoss: 0.125671 | BCEWithLogitsLoss: 0.5177
Epoch    140 | MSELoss: 0.121438 | BCEWithLogitsLoss: 0.5134
Epoch    150 | MSELoss: 0.116185 | BCEWithLogitsLoss: 0.5014
Epoch    160 | MSELoss: 0.110240 | BCEWithLogitsLoss: 0.5043
Epoch    170 | MSELoss: 0.112434 | BCEWithLogitsLoss: 0.5102
Epoch    180 | MSELoss: 0.106380 | BCEWithLogitsLoss: 0.5261
Epoch    190 | MSELoss: 0.086204 | BCEWithLogitsLoss: 0.4890
Epoch    200 | MSELoss: 0.089187 | BCEWithLogitsLoss: 0.4762
Epoch    210 | MSELoss: 0.087700 | BCEWithLogitsLoss: 0.4845
Epoch    220 | MSELoss: 0.084729 | BCEWithLogitsLoss: 0.4767
Epoch    230 | MSELoss: 0.085071 | BCEWithLogitsLoss: 0.4852
Epoch    240 | MSELoss: 0.089972 | BCEWithLogitsLoss: 0.5055
Epoch    250 | MSELoss: 0.086337 | BCEWithLogitsLoss: 0.4904
Epoch    260 | MSELoss: 0.087159 | BCEWithLogitsLoss: 0.4952
Epoch    270 | MSELoss: 0.083288 | BCEWithLogitsLoss: 0.4816
Epoch    280 | MSELoss: 0.077361 | BCEWithLogitsLoss: 0.4728
Epoch    290 | MSELoss: 0.082297 | BCEWithLogitsLoss: 0.4784
Epoch    300 | MSELoss: 0.092753 | BCEWithLogitsLoss: 0.4964
Epoch    310 | MSELoss: 0.081546 | BCEWithLogitsLoss: 0.4986
Epoch    320 | MSELoss: 0.082398 | BCEWithLogitsLoss: 0.5061
Epoch    330 | MSELoss: 0.082174 | BCEWithLogitsLoss: 0.4615
Epoch    340 | MSELoss: 0.087870 | BCEWithLogitsLoss: 0.4732
Epoch    350 | MSELoss: 0.077145 | BCEWithLogitsLoss: 0.4875
Epoch    360 | MSELoss: 0.074465 | BCEWithLogitsLoss: 0.4898
Epoch    370 | MSELoss: 0.075363 | BCEWithLogitsLoss: 0.4801
Epoch    380 | MSELoss: 0.077283 | BCEWithLogitsLoss: 0.4900
Epoch    390 | MSELoss: 0.076683 | BCEWithLogitsLoss: 0.4538
Epoch    400 | MSELoss: 0.073249 | BCEWithLogitsLoss: 0.4914
Epoch    410 | MSELoss: 0.079426 | BCEWithLogitsLoss: 0.4933
Epoch    420 | MSELoss: 0.077209 | BCEWithLogitsLoss: 0.4750
Epoch    430 | MSELoss: 0.077883 | BCEWithLogitsLoss: 0.4596
Epoch    440 | MSELoss: 0.075017 | BCEWithLogitsLoss: 0.4638
Epoch    450 | MSELoss: 0.075109 | BCEWithLogitsLoss: 0.4689
Epoch    460 | MSELoss: 0.077229 | BCEWithLogitsLoss: 0.4811
Epoch    470 | MSELoss: 0.069163 | BCEWithLogitsLoss: 0.4708
Epoch    480 | MSELoss: 0.070090 | BCEWithLogitsLoss: 0.4800
Epoch    490 | MSELoss: 0.075003 | BCEWithLogitsLoss: 0.4633
Epoch    500 | MSELoss: 0.078077 | BCEWithLogitsLoss: 0.4505
Epoch    510 | MSELoss: 0.069999 | BCEWithLogitsLoss: 0.4678
Epoch    520 | MSELoss: 0.068650 | BCEWithLogitsLoss: 0.4744
Epoch    530 | MSELoss: 0.069298 | BCEWithLogitsLoss: 0.4938
Epoch    540 | MSELoss: 0.071163 | BCEWithLogitsLoss: 0.4588
Epoch    550 | MSELoss: 0.067096 | BCEWithLogitsLoss: 0.4669
Epoch    560 | MSELoss: 0.069891 | BCEWithLogitsLoss: 0.4553
Epoch    570 | MSELoss: 0.071552 | BCEWithLogitsLoss: 0.4704
Epoch    580 | MSELoss: 0.074232 | BCEWithLogitsLoss: 0.4761
Epoch    590 | MSELoss: 0.073469 | BCEWithLogitsLoss: 0.4933
Epoch    600 | MSELoss: 0.075180 | BCEWithLogitsLoss: 0.4930
Epoch    610 | MSELoss: 0.074623 | BCEWithLogitsLoss: 0.4799
Epoch    620 | MSELoss: 0.070314 | BCEWithLogitsLoss: 0.4785
Epoch    630 | MSELoss: 0.067964 | BCEWithLogitsLoss: 0.4818
Epoch    640 | MSELoss: 0.075580 | BCEWithLogitsLoss: 0.4551
Epoch    650 | MSELoss: 0.070665 | BCEWithLogitsLoss: 0.4811
Epoch    660 | MSELoss: 0.076303 | BCEWithLogitsLoss: 0.4767
Epoch    670 | MSELoss: 0.077701 | BCEWithLogitsLoss: 0.4457
Epoch    680 | MSELoss: 0.067745 | BCEWithLogitsLoss: 0.4692
Epoch    690 | MSELoss: 0.069416 | BCEWithLogitsLoss: 0.4749
Epoch    700 | MSELoss: 0.062336 | BCEWithLogitsLoss: 0.4739
Epoch    710 | MSELoss: 0.068678 | BCEWithLogitsLoss: 0.4983
Epoch    720 | MSELoss: 0.073944 | BCEWithLogitsLoss: 0.5055
Epoch    730 | MSELoss: 0.073357 | BCEWithLogitsLoss: 0.4738
Epoch    740 | MSELoss: 0.064663 | BCEWithLogitsLoss: 0.4698
Epoch    750 | MSELoss: 0.064182 | BCEWithLogitsLoss: 0.4691
Epoch    760 | MSELoss: 0.077011 | BCEWithLogitsLoss: 0.4967
Epoch    770 | MSELoss: 0.071061 | BCEWithLogitsLoss: 0.4790
Epoch    780 | MSELoss: 0.064862 | BCEWithLogitsLoss: 0.4791
Epoch    790 | MSELoss: 0.067708 | BCEWithLogitsLoss: 0.4459
Epoch    800 | MSELoss: 0.066903 | BCEWithLogitsLoss: 0.4699
Epoch    810 | MSELoss: 0.061635 | BCEWithLogitsLoss: 0.4668
Epoch    820 | MSELoss: 0.072497 | BCEWithLogitsLoss: 0.4400
Epoch    830 | MSELoss: 0.080793 | BCEWithLogitsLoss: 0.4670
Epoch    840 | MSELoss: 0.066755 | BCEWithLogitsLoss: 0.4598
Epoch    850 | MSELoss: 0.064253 | BCEWithLogitsLoss: 0.4781
Epoch    860 | MSELoss: 0.064093 | BCEWithLogitsLoss: 0.4742
Epoch    870 | MSELoss: 0.065369 | BCEWithLogitsLoss: 0.4691
Epoch    880 | MSELoss: 0.064277 | BCEWithLogitsLoss: 0.4681
Epoch    890 | MSELoss: 0.067327 | BCEWithLogitsLoss: 0.4813
Epoch    900 | MSELoss: 0.065056 | BCEWithLogitsLoss: 0.4612
Epoch    910 | MSELoss: 0.068319 | BCEWithLogitsLoss: 0.4638
Epoch    920 | MSELoss: 0.069755 | BCEWithLogitsLoss: 0.4967
Epoch    930 | MSELoss: 0.065598 | BCEWithLogitsLoss: 0.4913
Epoch    940 | MSELoss: 0.069593 | BCEWithLogitsLoss: 0.4723
Epoch    950 | MSELoss: 0.066561 | BCEWithLogitsLoss: 0.4708
Epoch    960 | MSELoss: 0.073933 | BCEWithLogitsLoss: 0.4143
Epoch    970 | MSELoss: 0.060643 | BCEWithLogitsLoss: 0.4838
Epoch    980 | MSELoss: 0.069564 | BCEWithLogitsLoss: 0.4489
Epoch    990 | MSELoss: 0.060228 | BCEWithLogitsLoss: 0.4587
Epoch   1000 | MSELoss: 0.058726 | BCEWithLogitsLoss: 0.4635
Epoch   1010 | MSELoss: 0.058959 | BCEWithLogitsLoss: 0.4533
Epoch   1020 | MSELoss: 0.065538 | BCEWithLogitsLoss: 0.4289
Epoch   1030 | MSELoss: 0.061522 | BCEWithLogitsLoss: 0.4719
Epoch   1040 | MSELoss: 0.055120 | BCEWithLogitsLoss: 0.4353
Epoch   1050 | MSELoss: 0.058654 | BCEWithLogitsLoss: 0.4463
Epoch   1060 | MSELoss: 0.061210 | BCEWithLogitsLoss: 0.4811
Epoch   1070 | MSELoss: 0.062473 | BCEWithLogitsLoss: 0.4425
Epoch   1080 | MSELoss: 0.052549 | BCEWithLogitsLoss: 0.4752
Epoch   1090 | MSELoss: 0.059151 | BCEWithLogitsLoss: 0.4780
Epoch   1100 | MSELoss: 0.069705 | BCEWithLogitsLoss: 0.5064
Epoch   1110 | MSELoss: 0.064892 | BCEWithLogitsLoss: 0.4297
Epoch   1120 | MSELoss: 0.052827 | BCEWithLogitsLoss: 0.4470
Epoch   1130 | MSELoss: 0.061125 | BCEWithLogitsLoss: 0.4914
Epoch   1140 | MSELoss: 0.061076 | BCEWithLogitsLoss: 0.4532
Epoch   1150 | MSELoss: 0.052616 | BCEWithLogitsLoss: 0.4546
Epoch   1160 | MSELoss: 0.059846 | BCEWithLogitsLoss: 0.4810
Epoch   1170 | MSELoss: 0.057859 | BCEWithLogitsLoss: 0.4465
Epoch   1180 | MSELoss: 0.051245 | BCEWithLogitsLoss: 0.4630
Epoch   1190 | MSELoss: 0.058133 | BCEWithLogitsLoss: 0.4591
Epoch   1200 | MSELoss: 0.062021 | BCEWithLogitsLoss: 0.4741
Epoch   1210 | MSELoss: 0.056702 | BCEWithLogitsLoss: 0.4409
Epoch   1220 | MSELoss: 0.047237 | BCEWithLogitsLoss: 0.4586
Epoch   1230 | MSELoss: 0.048473 | BCEWithLogitsLoss: 0.4292
Epoch   1240 | MSELoss: 0.047951 | BCEWithLogitsLoss: 0.4749
Epoch   1250 | MSELoss: 0.050237 | BCEWithLogitsLoss: 0.4776
Epoch   1260 | MSELoss: 0.053342 | BCEWithLogitsLoss: 0.4906
Epoch   1270 | MSELoss: 0.042526 | BCEWithLogitsLoss: 0.4353
Epoch   1280 | MSELoss: 0.045365 | BCEWithLogitsLoss: 0.4471
Epoch   1290 | MSELoss: 0.045589 | BCEWithLogitsLoss: 0.4306
Epoch   1300 | MSELoss: 0.053523 | BCEWithLogitsLoss: 0.4312
Epoch   1310 | MSELoss: 0.041404 | BCEWithLogitsLoss: 0.4624
Epoch   1320 | MSELoss: 0.048252 | BCEWithLogitsLoss: 0.4817
Epoch   1330 | MSELoss: 0.040782 | BCEWithLogitsLoss: 0.4325
Epoch   1340 | MSELoss: 0.129960 | BCEWithLogitsLoss: 0.3582
Epoch   1350 | MSELoss: 0.062804 | BCEWithLogitsLoss: 0.4570
Epoch   1360 | MSELoss: 0.048720 | BCEWithLogitsLoss: 0.4425
Epoch   1370 | MSELoss: 0.054907 | BCEWithLogitsLoss: 0.4699
Epoch   1380 | MSELoss: 0.044143 | BCEWithLogitsLoss: 0.4487
Epoch   1390 | MSELoss: 0.035119 | BCEWithLogitsLoss: 0.4265
Epoch   1400 | MSELoss: 0.031863 | BCEWithLogitsLoss: 0.4334
Epoch   1410 | MSELoss: 0.032468 | BCEWithLogitsLoss: 0.4292
Epoch   1420 | MSELoss: 0.049294 | BCEWithLogitsLoss: 0.4108
Epoch   1430 | MSELoss: 0.051929 | BCEWithLogitsLoss: 0.3976
Epoch   1440 | MSELoss: 0.032112 | BCEWithLogitsLoss: 0.4521
Epoch   1450 | MSELoss: 0.032285 | BCEWithLogitsLoss: 0.4701
Epoch   1460 | MSELoss: 0.031933 | BCEWithLogitsLoss: 0.4319
Epoch   1470 | MSELoss: 0.026414 | BCEWithLogitsLoss: 0.4357
Epoch   1480 | MSELoss: 0.029551 | BCEWithLogitsLoss: 0.4684
Epoch   1490 | MSELoss: 0.027262 | BCEWithLogitsLoss: 0.4123
Epoch   1500 | MSELoss: 0.023771 | BCEWithLogitsLoss: 0.4367

------------------------------------------------------------
  DenseResNet_7x7_1
DenseResNet(
  (layers): ModuleList(
    (0): Sequential(
      (0): Linear(in_features=1, out_features=7, bias=True)
      (1): Mish()
    )
    (1-4): 4 x Sequential(
      (0): Linear(in_features=7, out_features=7, bias=True)
      (1): Mish()
    )
  )
  (head): Linear(in_features=7, out_features=1, bias=True)
)
------------------------------------------------------------

Epoch      1 | MSELoss: 0.586721 | BCEWithLogitsLoss: 0.9449
Epoch     10 | MSELoss: 0.375777 | BCEWithLogitsLoss: 0.7647
Epoch     20 | MSELoss: 0.363670 | BCEWithLogitsLoss: 0.6790
Epoch     30 | MSELoss: 0.349851 | BCEWithLogitsLoss: 0.7433
Epoch     40 | MSELoss: 0.336923 | BCEWithLogitsLoss: 0.6986
Epoch     50 | MSELoss: 0.323043 | BCEWithLogitsLoss: 0.7018
Epoch     60 | MSELoss: 0.307097 | BCEWithLogitsLoss: 0.6970
Epoch     70 | MSELoss: 0.292804 | BCEWithLogitsLoss: 0.6764
Epoch     80 | MSELoss: 0.284099 | BCEWithLogitsLoss: 0.6607
Epoch     90 | MSELoss: 0.276366 | BCEWithLogitsLoss: 0.6538
Epoch    100 | MSELoss: 0.267463 | BCEWithLogitsLoss: 0.6504
Epoch    110 | MSELoss: 0.255785 | BCEWithLogitsLoss: 0.6436
Epoch    120 | MSELoss: 0.241763 | BCEWithLogitsLoss: 0.6344
Epoch    130 | MSELoss: 0.228419 | BCEWithLogitsLoss: 0.6245
Epoch    140 | MSELoss: 0.215779 | BCEWithLogitsLoss: 0.6129
Epoch    150 | MSELoss: 0.201584 | BCEWithLogitsLoss: 0.6012
Epoch    160 | MSELoss: 0.182462 | BCEWithLogitsLoss: 0.5901
Epoch    170 | MSELoss: 0.157774 | BCEWithLogitsLoss: 0.5676
Epoch    180 | MSELoss: 0.127418 | BCEWithLogitsLoss: 0.5350
Epoch    190 | MSELoss: 0.104074 | BCEWithLogitsLoss: 0.5042
Epoch    200 | MSELoss: 0.092117 | BCEWithLogitsLoss: 0.4958
Epoch    210 | MSELoss: 0.090667 | BCEWithLogitsLoss: 0.4734
Epoch    220 | MSELoss: 0.085040 | BCEWithLogitsLoss: 0.4740
Epoch    230 | MSELoss: 0.079920 | BCEWithLogitsLoss: 0.4781
Epoch    240 | MSELoss: 0.075445 | BCEWithLogitsLoss: 0.4885
Epoch    250 | MSELoss: 0.069474 | BCEWithLogitsLoss: 0.4803
Epoch    260 | MSELoss: 0.063472 | BCEWithLogitsLoss: 0.4645
Epoch    270 | MSELoss: 0.058081 | BCEWithLogitsLoss: 0.4877
Epoch    280 | MSELoss: 0.051395 | BCEWithLogitsLoss: 0.4812
Epoch    290 | MSELoss: 0.045671 | BCEWithLogitsLoss: 0.4307
Epoch    300 | MSELoss: 0.038763 | BCEWithLogitsLoss: 0.4247
Epoch    310 | MSELoss: 0.031242 | BCEWithLogitsLoss: 0.4471
Epoch    320 | MSELoss: 0.029673 | BCEWithLogitsLoss: 0.4555
Epoch    330 | MSELoss: 0.026724 | BCEWithLogitsLoss: 0.4346
Epoch    340 | MSELoss: 0.026328 | BCEWithLogitsLoss: 0.4381
Epoch    350 | MSELoss: 0.025674 | BCEWithLogitsLoss: 0.4334
Epoch    360 | MSELoss: 0.025442 | BCEWithLogitsLoss: 0.4307
Epoch    370 | MSELoss: 0.025218 | BCEWithLogitsLoss: 0.4366
Epoch    380 | MSELoss: 0.024977 | BCEWithLogitsLoss: 0.4346
Epoch    390 | MSELoss: 0.024771 | BCEWithLogitsLoss: 0.4332
Epoch    400 | MSELoss: 0.024568 | BCEWithLogitsLoss: 0.4330
Epoch    410 | MSELoss: 0.024355 | BCEWithLogitsLoss: 0.4332
Epoch    420 | MSELoss: 0.024124 | BCEWithLogitsLoss: 0.4332
Epoch    430 | MSELoss: 0.023875 | BCEWithLogitsLoss: 0.4326
Epoch    440 | MSELoss: 0.023632 | BCEWithLogitsLoss: 0.4320
Epoch    450 | MSELoss: 0.026305 | BCEWithLogitsLoss: 0.4095
Epoch    460 | MSELoss: 0.023383 | BCEWithLogitsLoss: 0.4274
Epoch    470 | MSELoss: 0.023376 | BCEWithLogitsLoss: 0.4249
Epoch    480 | MSELoss: 0.023144 | BCEWithLogitsLoss: 0.4410
Epoch    490 | MSELoss: 0.022266 | BCEWithLogitsLoss: 0.4311
Epoch    500 | MSELoss: 0.021918 | BCEWithLogitsLoss: 0.4299
Epoch    510 | MSELoss: 0.021558 | BCEWithLogitsLoss: 0.4332
Epoch    520 | MSELoss: 0.021162 | BCEWithLogitsLoss: 0.4325
Epoch    530 | MSELoss: 0.020689 | BCEWithLogitsLoss: 0.4315
Epoch    540 | MSELoss: 0.020164 | BCEWithLogitsLoss: 0.4292
Epoch    550 | MSELoss: 0.045120 | BCEWithLogitsLoss: 0.3681
Epoch    560 | MSELoss: 0.024018 | BCEWithLogitsLoss: 0.4616
Epoch    570 | MSELoss: 0.020455 | BCEWithLogitsLoss: 0.4173
Epoch    580 | MSELoss: 0.018837 | BCEWithLogitsLoss: 0.4253
Epoch    590 | MSELoss: 0.018536 | BCEWithLogitsLoss: 0.4376
Epoch    600 | MSELoss: 0.017830 | BCEWithLogitsLoss: 0.4252
Epoch    610 | MSELoss: 0.017280 | BCEWithLogitsLoss: 0.4267
Epoch    620 | MSELoss: 0.016759 | BCEWithLogitsLoss: 0.4268
Epoch    630 | MSELoss: 0.019436 | BCEWithLogitsLoss: 0.4026
Epoch    640 | MSELoss: 0.018510 | BCEWithLogitsLoss: 0.4529
Epoch    650 | MSELoss: 0.017253 | BCEWithLogitsLoss: 0.4086
Epoch    660 | MSELoss: 0.015267 | BCEWithLogitsLoss: 0.4333
Epoch    670 | MSELoss: 0.017498 | BCEWithLogitsLoss: 0.4529
Epoch    680 | MSELoss: 0.014964 | BCEWithLogitsLoss: 0.4273
Epoch    690 | MSELoss: 0.014361 | BCEWithLogitsLoss: 0.4265
Epoch    700 | MSELoss: 0.021828 | BCEWithLogitsLoss: 0.3862
Epoch    710 | MSELoss: 0.014627 | BCEWithLogitsLoss: 0.4240
Epoch    720 | MSELoss: 0.014231 | BCEWithLogitsLoss: 0.4327
Epoch    730 | MSELoss: 0.013423 | BCEWithLogitsLoss: 0.4345
Epoch    740 | MSELoss: 0.012838 | BCEWithLogitsLoss: 0.4239
Epoch    750 | MSELoss: 0.025811 | BCEWithLogitsLoss: 0.3785
Epoch    760 | MSELoss: 0.015422 | BCEWithLogitsLoss: 0.4029
Epoch    770 | MSELoss: 0.014458 | BCEWithLogitsLoss: 0.4054
Epoch    780 | MSELoss: 0.012665 | BCEWithLogitsLoss: 0.4155
Epoch    790 | MSELoss: 0.012202 | BCEWithLogitsLoss: 0.4162
Epoch    800 | MSELoss: 0.012107 | BCEWithLogitsLoss: 0.4180
Epoch    810 | MSELoss: 0.011430 | BCEWithLogitsLoss: 0.4220
Epoch    820 | MSELoss: 0.011295 | BCEWithLogitsLoss: 0.4265
Epoch    830 | MSELoss: 0.022072 | BCEWithLogitsLoss: 0.4653
Epoch    840 | MSELoss: 0.012881 | BCEWithLogitsLoss: 0.4118
Epoch    850 | MSELoss: 0.014777 | BCEWithLogitsLoss: 0.3984
Epoch    860 | MSELoss: 0.011684 | BCEWithLogitsLoss: 0.4145
Epoch    870 | MSELoss: 0.011249 | BCEWithLogitsLoss: 0.4283
Epoch    880 | MSELoss: 0.010933 | BCEWithLogitsLoss: 0.4269
Epoch    890 | MSELoss: 0.010759 | BCEWithLogitsLoss: 0.4210
Epoch    900 | MSELoss: 0.010562 | BCEWithLogitsLoss: 0.4250
Epoch    910 | MSELoss: 0.010391 | BCEWithLogitsLoss: 0.4228
Epoch    920 | MSELoss: 0.010253 | BCEWithLogitsLoss: 0.4224
Epoch    930 | MSELoss: 0.010123 | BCEWithLogitsLoss: 0.4226
Epoch    940 | MSELoss: 0.010006 | BCEWithLogitsLoss: 0.4224
Epoch    950 | MSELoss: 0.009897 | BCEWithLogitsLoss: 0.4221
Epoch    960 | MSELoss: 0.010281 | BCEWithLogitsLoss: 0.4128
Epoch    970 | MSELoss: 0.023404 | BCEWithLogitsLoss: 0.4723
Epoch    980 | MSELoss: 0.011667 | BCEWithLogitsLoss: 0.4113
Epoch    990 | MSELoss: 0.010184 | BCEWithLogitsLoss: 0.4298
Epoch   1000 | MSELoss: 0.009889 | BCEWithLogitsLoss: 0.4168
Epoch   1010 | MSELoss: 0.009703 | BCEWithLogitsLoss: 0.4206
Epoch   1020 | MSELoss: 0.009663 | BCEWithLogitsLoss: 0.4257
Epoch   1030 | MSELoss: 0.009499 | BCEWithLogitsLoss: 0.4219
Epoch   1040 | MSELoss: 0.009417 | BCEWithLogitsLoss: 0.4210
Epoch   1050 | MSELoss: 0.009338 | BCEWithLogitsLoss: 0.4217
Epoch   1060 | MSELoss: 0.009268 | BCEWithLogitsLoss: 0.4219
Epoch   1070 | MSELoss: 0.009200 | BCEWithLogitsLoss: 0.4207
Epoch   1080 | MSELoss: 0.009147 | BCEWithLogitsLoss: 0.4195
Epoch   1090 | MSELoss: 0.012439 | BCEWithLogitsLoss: 0.3941
Epoch   1100 | MSELoss: 0.013547 | BCEWithLogitsLoss: 0.4507
Epoch   1110 | MSELoss: 0.009449 | BCEWithLogitsLoss: 0.4259
Epoch   1120 | MSELoss: 0.009345 | BCEWithLogitsLoss: 0.4128
Epoch   1130 | MSELoss: 0.009285 | BCEWithLogitsLoss: 0.4152
Epoch   1140 | MSELoss: 0.009044 | BCEWithLogitsLoss: 0.4167
Epoch   1150 | MSELoss: 0.008916 | BCEWithLogitsLoss: 0.4185
Epoch   1160 | MSELoss: 0.008839 | BCEWithLogitsLoss: 0.4211
Epoch   1170 | MSELoss: 0.008787 | BCEWithLogitsLoss: 0.4205
Epoch   1180 | MSELoss: 0.008748 | BCEWithLogitsLoss: 0.4215
Epoch   1190 | MSELoss: 0.008815 | BCEWithLogitsLoss: 0.4253
Epoch   1200 | MSELoss: 0.022545 | BCEWithLogitsLoss: 0.4748
Epoch   1210 | MSELoss: 0.011536 | BCEWithLogitsLoss: 0.4462
Epoch   1220 | MSELoss: 0.009195 | BCEWithLogitsLoss: 0.4305
Epoch   1230 | MSELoss: 0.008786 | BCEWithLogitsLoss: 0.4177
Epoch   1240 | MSELoss: 0.008754 | BCEWithLogitsLoss: 0.4164
Epoch   1250 | MSELoss: 0.008620 | BCEWithLogitsLoss: 0.4179
Epoch   1260 | MSELoss: 0.008545 | BCEWithLogitsLoss: 0.4190
Epoch   1270 | MSELoss: 0.008499 | BCEWithLogitsLoss: 0.4196
Epoch   1280 | MSELoss: 0.008459 | BCEWithLogitsLoss: 0.4195
Epoch   1290 | MSELoss: 0.008423 | BCEWithLogitsLoss: 0.4198
Epoch   1300 | MSELoss: 0.008392 | BCEWithLogitsLoss: 0.4198
Epoch   1310 | MSELoss: 0.008363 | BCEWithLogitsLoss: 0.4198
Epoch   1320 | MSELoss: 0.008747 | BCEWithLogitsLoss: 0.4289
Epoch   1330 | MSELoss: 0.011440 | BCEWithLogitsLoss: 0.3977
Epoch   1340 | MSELoss: 0.009365 | BCEWithLogitsLoss: 0.4117
Epoch   1350 | MSELoss: 0.008523 | BCEWithLogitsLoss: 0.4215
Epoch   1360 | MSELoss: 0.008635 | BCEWithLogitsLoss: 0.4247
Epoch   1370 | MSELoss: 0.008511 | BCEWithLogitsLoss: 0.4144
Epoch   1380 | MSELoss: 0.008391 | BCEWithLogitsLoss: 0.4230
Epoch   1390 | MSELoss: 0.008322 | BCEWithLogitsLoss: 0.4183
Epoch   1400 | MSELoss: 0.008271 | BCEWithLogitsLoss: 0.4199
Epoch   1410 | MSELoss: 0.008242 | BCEWithLogitsLoss: 0.4201
Epoch   1420 | MSELoss: 0.008214 | BCEWithLogitsLoss: 0.4191
Epoch   1430 | MSELoss: 0.008190 | BCEWithLogitsLoss: 0.4191
Epoch   1440 | MSELoss: 0.008168 | BCEWithLogitsLoss: 0.4192
Epoch   1450 | MSELoss: 0.008148 | BCEWithLogitsLoss: 0.4192
Epoch   1460 | MSELoss: 0.008130 | BCEWithLogitsLoss: 0.4192
Epoch   1470 | MSELoss: 0.008113 | BCEWithLogitsLoss: 0.4191
Epoch   1480 | MSELoss: 0.008097 | BCEWithLogitsLoss: 0.4191
Epoch   1490 | MSELoss: 0.008082 | BCEWithLogitsLoss: 0.4191
Epoch   1500 | MSELoss: 0.008068 | BCEWithLogitsLoss: 0.4190

------------------------------------------------------------
  ConvResNet_7x7_1
ConvResNet(
  (layers): ModuleList(
    (0): Sequential(
      (0): Conv1d(1, 7, kernel_size=(1,), stride=(1,))
      (1): Mish()
    )
    (1-4): 4 x Sequential(
      (0): Conv1d(7, 7, kernel_size=(3,), stride=(1,), padding=(1,))
      (1): Mish()
    )
  )
  (head): Conv1d(7, 1, kernel_size=(1,), stride=(1,))
)
------------------------------------------------------------

Epoch      1 | MSELoss: 0.443747 | BCEWithLogitsLoss: 0.6224
Epoch     10 | MSELoss: 0.366167 | BCEWithLogitsLoss: 0.6968
Epoch     20 | MSELoss: 0.357976 | BCEWithLogitsLoss: 0.7151
Epoch     30 | MSELoss: 0.335867 | BCEWithLogitsLoss: 0.7178
Epoch     40 | MSELoss: 0.299154 | BCEWithLogitsLoss: 0.6808
Epoch     50 | MSELoss: 0.263148 | BCEWithLogitsLoss: 0.6366
Epoch     60 | MSELoss: 0.247176 | BCEWithLogitsLoss: 0.6340
Epoch     70 | MSELoss: 0.254848 | BCEWithLogitsLoss: 0.6496
Epoch     80 | MSELoss: 0.232147 | BCEWithLogitsLoss: 0.6131
Epoch     90 | MSELoss: 0.189149 | BCEWithLogitsLoss: 0.5924
Epoch    100 | MSELoss: 0.155894 | BCEWithLogitsLoss: 0.5741
Epoch    110 | MSELoss: 0.146977 | BCEWithLogitsLoss: 0.5469
Epoch    120 | MSELoss: 0.141504 | BCEWithLogitsLoss: 0.5405
Epoch    130 | MSELoss: 0.157611 | BCEWithLogitsLoss: 0.5348
Epoch    140 | MSELoss: 0.134096 | BCEWithLogitsLoss: 0.5358
Epoch    150 | MSELoss: 0.137296 | BCEWithLogitsLoss: 0.5356
Epoch    160 | MSELoss: 0.126749 | BCEWithLogitsLoss: 0.5265
Epoch    170 | MSELoss: 0.121150 | BCEWithLogitsLoss: 0.5092
Epoch    180 | MSELoss: 0.135452 | BCEWithLogitsLoss: 0.5227
Epoch    190 | MSELoss: 0.120433 | BCEWithLogitsLoss: 0.5133
Epoch    200 | MSELoss: 0.115100 | BCEWithLogitsLoss: 0.5109
Epoch    210 | MSELoss: 0.105435 | BCEWithLogitsLoss: 0.5159
Epoch    220 | MSELoss: 0.108537 | BCEWithLogitsLoss: 0.4906
Epoch    230 | MSELoss: 0.105263 | BCEWithLogitsLoss: 0.5000
Epoch    240 | MSELoss: 0.098535 | BCEWithLogitsLoss: 0.5126
Epoch    250 | MSELoss: 0.102491 | BCEWithLogitsLoss: 0.4984
Epoch    260 | MSELoss: 0.096423 | BCEWithLogitsLoss: 0.5049
Epoch    270 | MSELoss: 0.094395 | BCEWithLogitsLoss: 0.4895
Epoch    280 | MSELoss: 0.085911 | BCEWithLogitsLoss: 0.5031
Epoch    290 | MSELoss: 0.079051 | BCEWithLogitsLoss: 0.4838
Epoch    300 | MSELoss: 0.082707 | BCEWithLogitsLoss: 0.4902
Epoch    310 | MSELoss: 0.081165 | BCEWithLogitsLoss: 0.4884
Epoch    320 | MSELoss: 0.078392 | BCEWithLogitsLoss: 0.4744
Epoch    330 | MSELoss: 0.076262 | BCEWithLogitsLoss: 0.4709
Epoch    340 | MSELoss: 0.068180 | BCEWithLogitsLoss: 0.4765
Epoch    350 | MSELoss: 0.061821 | BCEWithLogitsLoss: 0.4723
Epoch    360 | MSELoss: 0.062346 | BCEWithLogitsLoss: 0.4706
Epoch    370 | MSELoss: 0.061890 | BCEWithLogitsLoss: 0.4578
Epoch    380 | MSELoss: 0.062151 | BCEWithLogitsLoss: 0.4656
Epoch    390 | MSELoss: 0.067556 | BCEWithLogitsLoss: 0.4774
Epoch    400 | MSELoss: 0.063346 | BCEWithLogitsLoss: 0.4742
Epoch    410 | MSELoss: 0.064517 | BCEWithLogitsLoss: 0.4668
Epoch    420 | MSELoss: 0.059736 | BCEWithLogitsLoss: 0.4722
Epoch    430 | MSELoss: 0.058665 | BCEWithLogitsLoss: 0.4640
Epoch    440 | MSELoss: 0.058279 | BCEWithLogitsLoss: 0.4592
Epoch    450 | MSELoss: 0.057498 | BCEWithLogitsLoss: 0.4714
Epoch    460 | MSELoss: 0.066448 | BCEWithLogitsLoss: 0.4678
Epoch    470 | MSELoss: 0.056380 | BCEWithLogitsLoss: 0.4601
Epoch    480 | MSELoss: 0.056834 | BCEWithLogitsLoss: 0.4570
Epoch    490 | MSELoss: 0.050634 | BCEWithLogitsLoss: 0.4621
Epoch    500 | MSELoss: 0.053489 | BCEWithLogitsLoss: 0.4617
Epoch    510 | MSELoss: 0.054005 | BCEWithLogitsLoss: 0.4589
Epoch    520 | MSELoss: 0.054150 | BCEWithLogitsLoss: 0.4649
Epoch    530 | MSELoss: 0.050801 | BCEWithLogitsLoss: 0.4592
Epoch    540 | MSELoss: 0.053790 | BCEWithLogitsLoss: 0.4560
Epoch    550 | MSELoss: 0.050449 | BCEWithLogitsLoss: 0.4583
Epoch    560 | MSELoss: 0.050706 | BCEWithLogitsLoss: 0.4527
Epoch    570 | MSELoss: 0.053174 | BCEWithLogitsLoss: 0.4505
Epoch    580 | MSELoss: 0.047463 | BCEWithLogitsLoss: 0.4547
Epoch    590 | MSELoss: 0.056479 | BCEWithLogitsLoss: 0.4657
Epoch    600 | MSELoss: 0.049105 | BCEWithLogitsLoss: 0.4514
Epoch    610 | MSELoss: 0.049009 | BCEWithLogitsLoss: 0.4503
Epoch    620 | MSELoss: 0.048645 | BCEWithLogitsLoss: 0.4498
Epoch    630 | MSELoss: 0.047932 | BCEWithLogitsLoss: 0.4515
Epoch    640 | MSELoss: 0.047811 | BCEWithLogitsLoss: 0.4572
Epoch    650 | MSELoss: 0.047475 | BCEWithLogitsLoss: 0.4499
Epoch    660 | MSELoss: 0.051201 | BCEWithLogitsLoss: 0.4501
Epoch    670 | MSELoss: 0.049394 | BCEWithLogitsLoss: 0.4617
Epoch    680 | MSELoss: 0.051100 | BCEWithLogitsLoss: 0.4613
Epoch    690 | MSELoss: 0.047897 | BCEWithLogitsLoss: 0.4521
Epoch    700 | MSELoss: 0.046790 | BCEWithLogitsLoss: 0.4545
Epoch    710 | MSELoss: 0.045104 | BCEWithLogitsLoss: 0.4511
Epoch    720 | MSELoss: 0.046807 | BCEWithLogitsLoss: 0.4541
Epoch    730 | MSELoss: 0.045016 | BCEWithLogitsLoss: 0.4557
Epoch    740 | MSELoss: 0.046098 | BCEWithLogitsLoss: 0.4476
Epoch    750 | MSELoss: 0.051246 | BCEWithLogitsLoss: 0.4619
Epoch    760 | MSELoss: 0.045837 | BCEWithLogitsLoss: 0.4501
Epoch    770 | MSELoss: 0.047196 | BCEWithLogitsLoss: 0.4554
Epoch    780 | MSELoss: 0.046535 | BCEWithLogitsLoss: 0.4556
Epoch    790 | MSELoss: 0.043062 | BCEWithLogitsLoss: 0.4440
Epoch    800 | MSELoss: 0.045865 | BCEWithLogitsLoss: 0.4607
Epoch    810 | MSELoss: 0.049086 | BCEWithLogitsLoss: 0.4407
Epoch    820 | MSELoss: 0.046736 | BCEWithLogitsLoss: 0.4526
Epoch    830 | MSELoss: 0.050286 | BCEWithLogitsLoss: 0.4716
Epoch    840 | MSELoss: 0.042403 | BCEWithLogitsLoss: 0.4466
Epoch    850 | MSELoss: 0.041640 | BCEWithLogitsLoss: 0.4456
Epoch    860 | MSELoss: 0.043369 | BCEWithLogitsLoss: 0.4533
Epoch    870 | MSELoss: 0.041854 | BCEWithLogitsLoss: 0.4535
Epoch    880 | MSELoss: 0.044004 | BCEWithLogitsLoss: 0.4613
Epoch    890 | MSELoss: 0.047825 | BCEWithLogitsLoss: 0.4521
Epoch    900 | MSELoss: 0.046281 | BCEWithLogitsLoss: 0.4438
Epoch    910 | MSELoss: 0.046401 | BCEWithLogitsLoss: 0.4569
Epoch    920 | MSELoss: 0.044455 | BCEWithLogitsLoss: 0.4482
Epoch    930 | MSELoss: 0.043059 | BCEWithLogitsLoss: 0.4408
Epoch    940 | MSELoss: 0.042433 | BCEWithLogitsLoss: 0.4481
Epoch    950 | MSELoss: 0.038524 | BCEWithLogitsLoss: 0.4491
Epoch    960 | MSELoss: 0.038377 | BCEWithLogitsLoss: 0.4438
Epoch    970 | MSELoss: 0.041791 | BCEWithLogitsLoss: 0.4470
Epoch    980 | MSELoss: 0.046970 | BCEWithLogitsLoss: 0.4320
Epoch    990 | MSELoss: 0.048471 | BCEWithLogitsLoss: 0.4607
Epoch   1000 | MSELoss: 0.045041 | BCEWithLogitsLoss: 0.4592
Epoch   1010 | MSELoss: 0.044554 | BCEWithLogitsLoss: 0.4496
Epoch   1020 | MSELoss: 0.042508 | BCEWithLogitsLoss: 0.4519
Epoch   1030 | MSELoss: 0.042866 | BCEWithLogitsLoss: 0.4518
Epoch   1040 | MSELoss: 0.044150 | BCEWithLogitsLoss: 0.4595
Epoch   1050 | MSELoss: 0.041410 | BCEWithLogitsLoss: 0.4481
Epoch   1060 | MSELoss: 0.040752 | BCEWithLogitsLoss: 0.4463
Epoch   1070 | MSELoss: 0.036353 | BCEWithLogitsLoss: 0.4459
Epoch   1080 | MSELoss: 0.035810 | BCEWithLogitsLoss: 0.4492
Epoch   1090 | MSELoss: 0.035691 | BCEWithLogitsLoss: 0.4477
Epoch   1100 | MSELoss: 0.043450 | BCEWithLogitsLoss: 0.4529
Epoch   1110 | MSELoss: 0.039360 | BCEWithLogitsLoss: 0.4586
Epoch   1120 | MSELoss: 0.035574 | BCEWithLogitsLoss: 0.4338
Epoch   1130 | MSELoss: 0.043129 | BCEWithLogitsLoss: 0.4509
Epoch   1140 | MSELoss: 0.042628 | BCEWithLogitsLoss: 0.4618
Epoch   1150 | MSELoss: 0.044069 | BCEWithLogitsLoss: 0.4533
Epoch   1160 | MSELoss: 0.043812 | BCEWithLogitsLoss: 0.4491
Epoch   1170 | MSELoss: 0.038760 | BCEWithLogitsLoss: 0.4490
Epoch   1180 | MSELoss: 0.041727 | BCEWithLogitsLoss: 0.4448
Epoch   1190 | MSELoss: 0.042385 | BCEWithLogitsLoss: 0.4431
Epoch   1200 | MSELoss: 0.040440 | BCEWithLogitsLoss: 0.4495
Epoch   1210 | MSELoss: 0.038567 | BCEWithLogitsLoss: 0.4573
Epoch   1220 | MSELoss: 0.038084 | BCEWithLogitsLoss: 0.4472
Epoch   1230 | MSELoss: 0.039310 | BCEWithLogitsLoss: 0.4436
Epoch   1240 | MSELoss: 0.038515 | BCEWithLogitsLoss: 0.4421
Epoch   1250 | MSELoss: 0.037851 | BCEWithLogitsLoss: 0.4323
Epoch   1260 | MSELoss: 0.030858 | BCEWithLogitsLoss: 0.4438
Epoch   1270 | MSELoss: 0.029774 | BCEWithLogitsLoss: 0.4495
Epoch   1280 | MSELoss: 0.036408 | BCEWithLogitsLoss: 0.4562
Epoch   1290 | MSELoss: 0.052903 | BCEWithLogitsLoss: 0.4214
Epoch   1300 | MSELoss: 0.045576 | BCEWithLogitsLoss: 0.4722
Epoch   1310 | MSELoss: 0.051180 | BCEWithLogitsLoss: 0.4384
Epoch   1320 | MSELoss: 0.046176 | BCEWithLogitsLoss: 0.4386
Epoch   1330 | MSELoss: 0.043379 | BCEWithLogitsLoss: 0.4621
Epoch   1340 | MSELoss: 0.037041 | BCEWithLogitsLoss: 0.4385
Epoch   1350 | MSELoss: 0.038159 | BCEWithLogitsLoss: 0.4424
Epoch   1360 | MSELoss: 0.036217 | BCEWithLogitsLoss: 0.4489
Epoch   1370 | MSELoss: 0.036381 | BCEWithLogitsLoss: 0.4420
Epoch   1380 | MSELoss: 0.035269 | BCEWithLogitsLoss: 0.4419
Epoch   1390 | MSELoss: 0.038539 | BCEWithLogitsLoss: 0.4400
Epoch   1400 | MSELoss: 0.035402 | BCEWithLogitsLoss: 0.4473
Epoch   1410 | MSELoss: 0.035095 | BCEWithLogitsLoss: 0.4480
Epoch   1420 | MSELoss: 0.035108 | BCEWithLogitsLoss: 0.4429
Epoch   1430 | MSELoss: 0.036584 | BCEWithLogitsLoss: 0.4484
Epoch   1440 | MSELoss: 0.036166 | BCEWithLogitsLoss: 0.4392
Epoch   1450 | MSELoss: 0.033453 | BCEWithLogitsLoss: 0.4440
Epoch   1460 | MSELoss: 0.035297 | BCEWithLogitsLoss: 0.4461
Epoch   1470 | MSELoss: 0.035115 | BCEWithLogitsLoss: 0.4450
Epoch   1480 | MSELoss: 0.034642 | BCEWithLogitsLoss: 0.4395
Epoch   1490 | MSELoss: 0.035901 | BCEWithLogitsLoss: 0.4484
Epoch   1500 | MSELoss: 0.034470 | BCEWithLogitsLoss: 0.4476

------------------------------------------------------------
  CustomNet_7x7_1
CustomNet(
  (_linears): ModuleList(
    (0): Linear(in_features=1, out_features=7, bias=True)
    (1-4): 4 x Linear(in_features=7, out_features=7, bias=True)
    (5): Linear(in_features=7, out_features=1, bias=True)
  )
  (_funcs): ModuleList(
    (0-4): 5 x Mish()
    (5): Identity()
  )
)
------------------------------------------------------------

Epoch      1 | MSELoss: 0.502001 | BCEWithLogitsLoss: 0.8758
Epoch     10 | MSELoss: 0.395121 | BCEWithLogitsLoss: 0.6550
Epoch     20 | MSELoss: 0.360672 | BCEWithLogitsLoss: 0.7378
Epoch     30 | MSELoss: 0.344783 | BCEWithLogitsLoss: 0.7037
Epoch     40 | MSELoss: 0.335686 | BCEWithLogitsLoss: 0.7145
Epoch     50 | MSELoss: 0.328047 | BCEWithLogitsLoss: 0.7054
Epoch     60 | MSELoss: 0.320015 | BCEWithLogitsLoss: 0.7054
Epoch     70 | MSELoss: 0.311831 | BCEWithLogitsLoss: 0.6931
Epoch     80 | MSELoss: 0.304148 | BCEWithLogitsLoss: 0.6901
Epoch     90 | MSELoss: 0.297760 | BCEWithLogitsLoss: 0.6796
Epoch    100 | MSELoss: 0.292591 | BCEWithLogitsLoss: 0.6741
Epoch    110 | MSELoss: 0.288051 | BCEWithLogitsLoss: 0.6701
Epoch    120 | MSELoss: 0.284164 | BCEWithLogitsLoss: 0.6666
Epoch    130 | MSELoss: 0.280940 | BCEWithLogitsLoss: 0.6629
Epoch    140 | MSELoss: 0.278296 | BCEWithLogitsLoss: 0.6589
Epoch    150 | MSELoss: 0.276107 | BCEWithLogitsLoss: 0.6556
Epoch    160 | MSELoss: 0.274122 | BCEWithLogitsLoss: 0.6527
Epoch    170 | MSELoss: 0.271990 | BCEWithLogitsLoss: 0.6498
Epoch    180 | MSELoss: 0.268732 | BCEWithLogitsLoss: 0.6452
Epoch    190 | MSELoss: 0.263314 | BCEWithLogitsLoss: 0.6368
Epoch    200 | MSELoss: 0.254913 | BCEWithLogitsLoss: 0.6245
Epoch    210 | MSELoss: 0.234540 | BCEWithLogitsLoss: 0.6112
Epoch    220 | MSELoss: 0.170563 | BCEWithLogitsLoss: 0.5712
Epoch    230 | MSELoss: 0.133030 | BCEWithLogitsLoss: 0.5189
Epoch    240 | MSELoss: 0.109315 | BCEWithLogitsLoss: 0.5071
Epoch    250 | MSELoss: 0.084775 | BCEWithLogitsLoss: 0.4976
Epoch    260 | MSELoss: 0.073423 | BCEWithLogitsLoss: 0.4612
Epoch    270 | MSELoss: 0.068254 | BCEWithLogitsLoss: 0.4556
Epoch    280 | MSELoss: 0.062568 | BCEWithLogitsLoss: 0.4686
Epoch    290 | MSELoss: 0.057881 | BCEWithLogitsLoss: 0.4649
Epoch    300 | MSELoss: 0.054599 | BCEWithLogitsLoss: 0.4525
Epoch    310 | MSELoss: 0.050533 | BCEWithLogitsLoss: 0.4682
Epoch    320 | MSELoss: 0.048750 | BCEWithLogitsLoss: 0.4365
Epoch    330 | MSELoss: 0.043122 | BCEWithLogitsLoss: 0.4418
Epoch    340 | MSELoss: 0.038879 | BCEWithLogitsLoss: 0.4506
Epoch    350 | MSELoss: 0.036634 | BCEWithLogitsLoss: 0.4365
Epoch    360 | MSELoss: 0.033652 | BCEWithLogitsLoss: 0.4364
Epoch    370 | MSELoss: 0.035725 | BCEWithLogitsLoss: 0.4212
Epoch    380 | MSELoss: 0.030116 | BCEWithLogitsLoss: 0.4480
Epoch    390 | MSELoss: 0.029266 | BCEWithLogitsLoss: 0.4505
Epoch    400 | MSELoss: 0.027345 | BCEWithLogitsLoss: 0.4401
Epoch    410 | MSELoss: 0.026070 | BCEWithLogitsLoss: 0.4356
Epoch    420 | MSELoss: 0.024958 | BCEWithLogitsLoss: 0.4352
Epoch    430 | MSELoss: 0.023848 | BCEWithLogitsLoss: 0.4333
Epoch    440 | MSELoss: 0.022816 | BCEWithLogitsLoss: 0.4351
Epoch    450 | MSELoss: 0.031981 | BCEWithLogitsLoss: 0.4633
Epoch    460 | MSELoss: 0.021591 | BCEWithLogitsLoss: 0.4381
Epoch    470 | MSELoss: 0.022432 | BCEWithLogitsLoss: 0.4461
Epoch    480 | MSELoss: 0.020255 | BCEWithLogitsLoss: 0.4311
Epoch    490 | MSELoss: 0.019495 | BCEWithLogitsLoss: 0.4276
Epoch    500 | MSELoss: 0.018856 | BCEWithLogitsLoss: 0.4324
Epoch    510 | MSELoss: 0.018286 | BCEWithLogitsLoss: 0.4310
Epoch    520 | MSELoss: 0.017800 | BCEWithLogitsLoss: 0.4284
Epoch    530 | MSELoss: 0.017337 | BCEWithLogitsLoss: 0.4295
Epoch    540 | MSELoss: 0.016925 | BCEWithLogitsLoss: 0.4285
Epoch    550 | MSELoss: 0.016549 | BCEWithLogitsLoss: 0.4284
Epoch    560 | MSELoss: 0.016209 | BCEWithLogitsLoss: 0.4279
Epoch    570 | MSELoss: 0.016446 | BCEWithLogitsLoss: 0.4236
Epoch    580 | MSELoss: 0.024219 | BCEWithLogitsLoss: 0.4565
Epoch    590 | MSELoss: 0.016182 | BCEWithLogitsLoss: 0.4205
Epoch    600 | MSELoss: 0.015385 | BCEWithLogitsLoss: 0.4259
Epoch    610 | MSELoss: 0.015320 | BCEWithLogitsLoss: 0.4312
Epoch    620 | MSELoss: 0.015110 | BCEWithLogitsLoss: 0.4251
Epoch    630 | MSELoss: 0.014815 | BCEWithLogitsLoss: 0.4256
Epoch    640 | MSELoss: 0.014608 | BCEWithLogitsLoss: 0.4268
Epoch    650 | MSELoss: 0.014431 | BCEWithLogitsLoss: 0.4259
Epoch    660 | MSELoss: 0.014253 | BCEWithLogitsLoss: 0.4262
Epoch    670 | MSELoss: 0.014083 | BCEWithLogitsLoss: 0.4265
Epoch    680 | MSELoss: 0.013929 | BCEWithLogitsLoss: 0.4267
Epoch    690 | MSELoss: 0.016298 | BCEWithLogitsLoss: 0.4372
Epoch    700 | MSELoss: 0.018657 | BCEWithLogitsLoss: 0.4073
Epoch    710 | MSELoss: 0.014322 | BCEWithLogitsLoss: 0.4340
Epoch    720 | MSELoss: 0.013985 | BCEWithLogitsLoss: 0.4299
Epoch    730 | MSELoss: 0.013527 | BCEWithLogitsLoss: 0.4237
Epoch    740 | MSELoss: 0.013392 | BCEWithLogitsLoss: 0.4223
Epoch    750 | MSELoss: 0.013184 | BCEWithLogitsLoss: 0.4244
Epoch    760 | MSELoss: 0.013055 | BCEWithLogitsLoss: 0.4242
Epoch    770 | MSELoss: 0.012936 | BCEWithLogitsLoss: 0.4242
Epoch    780 | MSELoss: 0.012823 | BCEWithLogitsLoss: 0.4249
Epoch    790 | MSELoss: 0.012717 | BCEWithLogitsLoss: 0.4244
Epoch    800 | MSELoss: 0.012617 | BCEWithLogitsLoss: 0.4247
Epoch    810 | MSELoss: 0.012549 | BCEWithLogitsLoss: 0.4254
Epoch    820 | MSELoss: 0.021425 | BCEWithLogitsLoss: 0.4455
Epoch    830 | MSELoss: 0.012486 | BCEWithLogitsLoss: 0.4237
Epoch    840 | MSELoss: 0.013846 | BCEWithLogitsLoss: 0.4359
Epoch    850 | MSELoss: 0.012709 | BCEWithLogitsLoss: 0.4292
Epoch    860 | MSELoss: 0.012242 | BCEWithLogitsLoss: 0.4219
Epoch    870 | MSELoss: 0.012167 | BCEWithLogitsLoss: 0.4217
Epoch    880 | MSELoss: 0.012074 | BCEWithLogitsLoss: 0.4228
Epoch    890 | MSELoss: 0.011997 | BCEWithLogitsLoss: 0.4233
Epoch    900 | MSELoss: 0.011934 | BCEWithLogitsLoss: 0.4235
Epoch    910 | MSELoss: 0.011864 | BCEWithLogitsLoss: 0.4232
Epoch    920 | MSELoss: 0.011801 | BCEWithLogitsLoss: 0.4233
Epoch    930 | MSELoss: 0.011740 | BCEWithLogitsLoss: 0.4233
Epoch    940 | MSELoss: 0.011682 | BCEWithLogitsLoss: 0.4232
Epoch    950 | MSELoss: 0.011643 | BCEWithLogitsLoss: 0.4232
Epoch    960 | MSELoss: 0.015626 | BCEWithLogitsLoss: 0.4308
Epoch    970 | MSELoss: 0.018185 | BCEWithLogitsLoss: 0.3958
Epoch    980 | MSELoss: 0.013456 | BCEWithLogitsLoss: 0.4083
Epoch    990 | MSELoss: 0.011967 | BCEWithLogitsLoss: 0.4214
Epoch   1000 | MSELoss: 0.011597 | BCEWithLogitsLoss: 0.4237
Epoch   1010 | MSELoss: 0.011438 | BCEWithLogitsLoss: 0.4211
Epoch   1020 | MSELoss: 0.011393 | BCEWithLogitsLoss: 0.4204
Epoch   1030 | MSELoss: 0.011328 | BCEWithLogitsLoss: 0.4213
Epoch   1040 | MSELoss: 0.011278 | BCEWithLogitsLoss: 0.4225
Epoch   1050 | MSELoss: 0.011238 | BCEWithLogitsLoss: 0.4225
Epoch   1060 | MSELoss: 0.011200 | BCEWithLogitsLoss: 0.4221
Epoch   1070 | MSELoss: 0.011163 | BCEWithLogitsLoss: 0.4224
Epoch   1080 | MSELoss: 0.011127 | BCEWithLogitsLoss: 0.4223
Epoch   1090 | MSELoss: 0.011092 | BCEWithLogitsLoss: 0.4222
Epoch   1100 | MSELoss: 0.011058 | BCEWithLogitsLoss: 0.4220
Epoch   1110 | MSELoss: 0.011138 | BCEWithLogitsLoss: 0.4194
Epoch   1120 | MSELoss: 0.031260 | BCEWithLogitsLoss: 0.3886
Epoch   1130 | MSELoss: 0.015469 | BCEWithLogitsLoss: 0.4064
Epoch   1140 | MSELoss: 0.011303 | BCEWithLogitsLoss: 0.4282
Epoch   1150 | MSELoss: 0.011455 | BCEWithLogitsLoss: 0.4297
Epoch   1160 | MSELoss: 0.011030 | BCEWithLogitsLoss: 0.4199
Epoch   1170 | MSELoss: 0.010983 | BCEWithLogitsLoss: 0.4202
Epoch   1180 | MSELoss: 0.010888 | BCEWithLogitsLoss: 0.4222
Epoch   1190 | MSELoss: 0.010869 | BCEWithLogitsLoss: 0.4228
Epoch   1200 | MSELoss: 0.010840 | BCEWithLogitsLoss: 0.4226
Epoch   1210 | MSELoss: 0.010813 | BCEWithLogitsLoss: 0.4222
Epoch   1220 | MSELoss: 0.010787 | BCEWithLogitsLoss: 0.4220
Epoch   1230 | MSELoss: 0.010763 | BCEWithLogitsLoss: 0.4219
Epoch   1240 | MSELoss: 0.010740 | BCEWithLogitsLoss: 0.4218
Epoch   1250 | MSELoss: 0.010718 | BCEWithLogitsLoss: 0.4217
Epoch   1260 | MSELoss: 0.010696 | BCEWithLogitsLoss: 0.4217
Epoch   1270 | MSELoss: 0.010674 | BCEWithLogitsLoss: 0.4216
Epoch   1280 | MSELoss: 0.010653 | BCEWithLogitsLoss: 0.4216
Epoch   1290 | MSELoss: 0.010632 | BCEWithLogitsLoss: 0.4215
Epoch   1300 | MSELoss: 0.010612 | BCEWithLogitsLoss: 0.4214
Epoch   1310 | MSELoss: 0.010598 | BCEWithLogitsLoss: 0.4205
Epoch   1320 | MSELoss: 0.012418 | BCEWithLogitsLoss: 0.4040
Epoch   1330 | MSELoss: 0.014097 | BCEWithLogitsLoss: 0.3984
Epoch   1340 | MSELoss: 0.011780 | BCEWithLogitsLoss: 0.4274
Epoch   1350 | MSELoss: 0.010780 | BCEWithLogitsLoss: 0.4157
Epoch   1360 | MSELoss: 0.010588 | BCEWithLogitsLoss: 0.4253
Epoch   1370 | MSELoss: 0.010541 | BCEWithLogitsLoss: 0.4202
Epoch   1380 | MSELoss: 0.010512 | BCEWithLogitsLoss: 0.4235
Epoch   1390 | MSELoss: 0.010460 | BCEWithLogitsLoss: 0.4214
Epoch   1400 | MSELoss: 0.010451 | BCEWithLogitsLoss: 0.4202
Epoch   1410 | MSELoss: 0.010629 | BCEWithLogitsLoss: 0.4164
Epoch   1420 | MSELoss: 0.018089 | BCEWithLogitsLoss: 0.3929
Epoch   1430 | MSELoss: 0.012028 | BCEWithLogitsLoss: 0.4349
Epoch   1440 | MSELoss: 0.011075 | BCEWithLogitsLoss: 0.4298
Epoch   1450 | MSELoss: 0.010380 | BCEWithLogitsLoss: 0.4212
Epoch   1460 | MSELoss: 0.010476 | BCEWithLogitsLoss: 0.4176
Epoch   1470 | MSELoss: 0.010352 | BCEWithLogitsLoss: 0.4207
Epoch   1480 | MSELoss: 0.010349 | BCEWithLogitsLoss: 0.4227
Epoch   1490 | MSELoss: 0.010320 | BCEWithLogitsLoss: 0.4206
Epoch   1500 | MSELoss: 0.010302 | BCEWithLogitsLoss: 0.4214
FCNN_7x7_1.gif ... saved
CNN_7x7_1.gif ... saved
DenseResNet_7x7_1.gif ... saved
ConvResNet_7x7_1.gif ... saved
CustomNet_7x7_1.gif ... saved
```
