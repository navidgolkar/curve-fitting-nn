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
- Activation function Mish() works best for all configurations
- MSELoss(), L1Loss(), HuberLoss(), SmoothL1Loss(), which have similar formulas, work best for this problem
- FCNN, albeit its dense configuration, converges best throughout all different hyperparameters

## To-do
- add a heuristic approach for optimizing a Neural Network (CustomNet) by pruning connections e.g.: Grey Wolves Optimization, Genetic Algorithm, etc.

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

Epoch      1 | MSELoss: 0.372075 | BCEWithLogitsLoss: 0.6859
Epoch     10 | MSELoss: 0.356548 | BCEWithLogitsLoss: 0.7144
Epoch     20 | MSELoss: 0.348612 | BCEWithLogitsLoss: 0.7277
Epoch     30 | MSELoss: 0.340448 | BCEWithLogitsLoss: 0.7223
Epoch     40 | MSELoss: 0.325578 | BCEWithLogitsLoss: 0.7099
Epoch     50 | MSELoss: 0.305226 | BCEWithLogitsLoss: 0.6907
Epoch     60 | MSELoss: 0.293911 | BCEWithLogitsLoss: 0.6643
Epoch     70 | MSELoss: 0.288561 | BCEWithLogitsLoss: 0.6630
Epoch     80 | MSELoss: 0.284188 | BCEWithLogitsLoss: 0.6621
Epoch     90 | MSELoss: 0.279031 | BCEWithLogitsLoss: 0.6543
Epoch    100 | MSELoss: 0.271343 | BCEWithLogitsLoss: 0.6473
Epoch    110 | MSELoss: 0.256411 | BCEWithLogitsLoss: 0.6374
Epoch    120 | MSELoss: 0.220392 | BCEWithLogitsLoss: 0.6089
Epoch    130 | MSELoss: 0.172852 | BCEWithLogitsLoss: 0.5661
Epoch    140 | MSELoss: 0.152462 | BCEWithLogitsLoss: 0.5432
Epoch    150 | MSELoss: 0.129620 | BCEWithLogitsLoss: 0.5160
Epoch    160 | MSELoss: 0.109254 | BCEWithLogitsLoss: 0.5037
Epoch    170 | MSELoss: 0.095110 | BCEWithLogitsLoss: 0.4985
Epoch    180 | MSELoss: 0.087569 | BCEWithLogitsLoss: 0.4861
Epoch    190 | MSELoss: 0.082191 | BCEWithLogitsLoss: 0.5019
Epoch    200 | MSELoss: 0.075966 | BCEWithLogitsLoss: 0.4756
Epoch    210 | MSELoss: 0.071449 | BCEWithLogitsLoss: 0.4741
Epoch    220 | MSELoss: 0.066833 | BCEWithLogitsLoss: 0.4749
Epoch    230 | MSELoss: 0.061446 | BCEWithLogitsLoss: 0.4707
Epoch    240 | MSELoss: 0.053157 | BCEWithLogitsLoss: 0.4618
Epoch    250 | MSELoss: 0.045225 | BCEWithLogitsLoss: 0.4677
Epoch    260 | MSELoss: 0.036979 | BCEWithLogitsLoss: 0.4555
Epoch    270 | MSELoss: 0.034399 | BCEWithLogitsLoss: 0.4491
Epoch    280 | MSELoss: 0.028673 | BCEWithLogitsLoss: 0.4471
Epoch    290 | MSELoss: 0.023769 | BCEWithLogitsLoss: 0.4435
Epoch    300 | MSELoss: 0.020465 | BCEWithLogitsLoss: 0.4380
Epoch    310 | MSELoss: 0.017677 | BCEWithLogitsLoss: 0.4278
Epoch    320 | MSELoss: 0.015165 | BCEWithLogitsLoss: 0.4275
Epoch    330 | MSELoss: 0.013438 | BCEWithLogitsLoss: 0.4245
Epoch    340 | MSELoss: 0.013903 | BCEWithLogitsLoss: 0.4329
Epoch    350 | MSELoss: 0.012269 | BCEWithLogitsLoss: 0.4158
Epoch    360 | MSELoss: 0.011562 | BCEWithLogitsLoss: 0.4178
Epoch    370 | MSELoss: 0.011269 | BCEWithLogitsLoss: 0.4275
Epoch    380 | MSELoss: 0.012266 | BCEWithLogitsLoss: 0.4388
Epoch    390 | MSELoss: 0.011114 | BCEWithLogitsLoss: 0.4321
Epoch    400 | MSELoss: 0.010650 | BCEWithLogitsLoss: 0.4245
Epoch    410 | MSELoss: 0.011926 | BCEWithLogitsLoss: 0.4125
Epoch    420 | MSELoss: 0.011726 | BCEWithLogitsLoss: 0.4140
Epoch    430 | MSELoss: 0.010696 | BCEWithLogitsLoss: 0.4175
Epoch    440 | MSELoss: 0.010173 | BCEWithLogitsLoss: 0.4224
Epoch    450 | MSELoss: 0.010085 | BCEWithLogitsLoss: 0.4192
Epoch    460 | MSELoss: 0.009978 | BCEWithLogitsLoss: 0.4180
Epoch    470 | MSELoss: 0.014816 | BCEWithLogitsLoss: 0.4036
Epoch    480 | MSELoss: 0.010680 | BCEWithLogitsLoss: 0.4099
Epoch    490 | MSELoss: 0.010326 | BCEWithLogitsLoss: 0.4284
Epoch    500 | MSELoss: 0.009812 | BCEWithLogitsLoss: 0.4185
Epoch    510 | MSELoss: 0.009800 | BCEWithLogitsLoss: 0.4170
Epoch    520 | MSELoss: 0.009694 | BCEWithLogitsLoss: 0.4191
Epoch    530 | MSELoss: 0.009656 | BCEWithLogitsLoss: 0.4198
Epoch    540 | MSELoss: 0.009703 | BCEWithLogitsLoss: 0.4205
Epoch    550 | MSELoss: 0.012582 | BCEWithLogitsLoss: 0.4192
Epoch    560 | MSELoss: 0.010617 | BCEWithLogitsLoss: 0.4119
Epoch    570 | MSELoss: 0.009832 | BCEWithLogitsLoss: 0.4144
Epoch    580 | MSELoss: 0.009588 | BCEWithLogitsLoss: 0.4219
Epoch    590 | MSELoss: 0.009575 | BCEWithLogitsLoss: 0.4230
Epoch    600 | MSELoss: 0.009509 | BCEWithLogitsLoss: 0.4221
Epoch    610 | MSELoss: 0.009477 | BCEWithLogitsLoss: 0.4213
Epoch    620 | MSELoss: 0.009449 | BCEWithLogitsLoss: 0.4200
Epoch    630 | MSELoss: 0.009429 | BCEWithLogitsLoss: 0.4195
Epoch    640 | MSELoss: 0.009474 | BCEWithLogitsLoss: 0.4194
Epoch    650 | MSELoss: 0.015290 | BCEWithLogitsLoss: 0.4180
Epoch    660 | MSELoss: 0.010199 | BCEWithLogitsLoss: 0.4266
Epoch    670 | MSELoss: 0.009876 | BCEWithLogitsLoss: 0.4296
Epoch    680 | MSELoss: 0.009403 | BCEWithLogitsLoss: 0.4183
Epoch    690 | MSELoss: 0.009439 | BCEWithLogitsLoss: 0.4160
Epoch    700 | MSELoss: 0.009360 | BCEWithLogitsLoss: 0.4196
Epoch    710 | MSELoss: 0.009348 | BCEWithLogitsLoss: 0.4209
Epoch    720 | MSELoss: 0.009334 | BCEWithLogitsLoss: 0.4207
Epoch    730 | MSELoss: 0.009319 | BCEWithLogitsLoss: 0.4204
Epoch    740 | MSELoss: 0.009305 | BCEWithLogitsLoss: 0.4199
Epoch    750 | MSELoss: 0.009294 | BCEWithLogitsLoss: 0.4196
Epoch    760 | MSELoss: 0.009282 | BCEWithLogitsLoss: 0.4198
Epoch    770 | MSELoss: 0.009272 | BCEWithLogitsLoss: 0.4198
Epoch    780 | MSELoss: 0.009283 | BCEWithLogitsLoss: 0.4198
Epoch    790 | MSELoss: 0.016015 | BCEWithLogitsLoss: 0.4235
Epoch    800 | MSELoss: 0.010609 | BCEWithLogitsLoss: 0.4203
Epoch    810 | MSELoss: 0.009985 | BCEWithLogitsLoss: 0.4204
Epoch    820 | MSELoss: 0.009280 | BCEWithLogitsLoss: 0.4217
Epoch    830 | MSELoss: 0.009359 | BCEWithLogitsLoss: 0.4208
Epoch    840 | MSELoss: 0.009270 | BCEWithLogitsLoss: 0.4201
Epoch    850 | MSELoss: 0.009233 | BCEWithLogitsLoss: 0.4199
Epoch    860 | MSELoss: 0.009220 | BCEWithLogitsLoss: 0.4197
Epoch    870 | MSELoss: 0.009212 | BCEWithLogitsLoss: 0.4197
Epoch    880 | MSELoss: 0.009203 | BCEWithLogitsLoss: 0.4198
Epoch    890 | MSELoss: 0.009195 | BCEWithLogitsLoss: 0.4199
Epoch    900 | MSELoss: 0.009187 | BCEWithLogitsLoss: 0.4196
Epoch    910 | MSELoss: 0.009180 | BCEWithLogitsLoss: 0.4196
Epoch    920 | MSELoss: 0.009173 | BCEWithLogitsLoss: 0.4197
Epoch    930 | MSELoss: 0.009166 | BCEWithLogitsLoss: 0.4196
Epoch    940 | MSELoss: 0.009159 | BCEWithLogitsLoss: 0.4197
Epoch    950 | MSELoss: 0.009157 | BCEWithLogitsLoss: 0.4203
Epoch    960 | MSELoss: 0.011533 | BCEWithLogitsLoss: 0.4352
Epoch    970 | MSELoss: 0.009363 | BCEWithLogitsLoss: 0.4170
Epoch    980 | MSELoss: 0.010214 | BCEWithLogitsLoss: 0.4287
Epoch    990 | MSELoss: 0.009364 | BCEWithLogitsLoss: 0.4137
Epoch   1000 | MSELoss: 0.009176 | BCEWithLogitsLoss: 0.4223
FCNN_7x7_1.gif ... saved
CNN_7x7_1.gif ... saved
DenseResNet_7x7_1.gif ... saved
ConvResNet_7x7_1.gif ... saved
CustomNet_7x7_1.gif ... saved
```
