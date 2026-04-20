# curve-fitting-nn
4 different configurations of neural networks used for curve fitting + visualization

in bash using arguments you can change:
-  --hn: Number of hidden layers [type=int, default=5]
-  --nn: number of nodes at each hidden layer (the number of layers is equal at all hidden layers) [type=int, default=7]
-  --func: Enter a number in the range (1-27) to select which activation function to use [type=int, default=17] (to see which number corresponds to what activation function, check paramters.py, 17 is Mish())
-  --loss1: Enter a number in the range (1-9) which loss unction to use for the training of models [type=int, default=2] (to see which number corresponds to what loss function, check paramters.py, 2 is Mean Squared Error[MSELoss()] and 6 is Binary Cross Entropy with Logits[BCEWithLogits()])
-  --loss2: Enter a number in the range (1-9) which loss unction to use for the second plot (this is not used for training) [type=int, default=6]
-  --lr: Enter learning rate value [type=float, default=1e-2]
-  --epoch: Number of epochs to run [type=int, default=1000]
-  --log: the results should should per how many epochs [type=int, default=10]
-  --grad_clip: Enter the value at which the gradient should be clipped to prevent explosion [type=float, default=100]
-  --seed: Seed number for random values [type=int, default=1]
-  --shuffle: Will shuffle input data of models [action='store_true']
-  --device: What device to use for pytorch [type=str, default="cpu"]
-  --verbose: Whether to show results in console [action='store_true']
-  --in_n: Number of input data [type=int, default=200]
-  --in_std: Standard deviation for input noise [type=float, default=1e-1]
-  --show: Whether to open figure files after running the code [action='store_true']
-  --file_type: What should be the file_type of saved figures (gif, png, jpeg) [type=str, default="gif"]
-  --conv: (kernel size, padding, stride) for convolutional neural network [type=tuple[int, int, int], default=(3, 1, 1)]
-  --connect: Number of connections for ConvResNet residual connections (connections start from layer+2) [type=int, default=1]
-  --name: added string at the end of each file for keeping track at running multiple runs [type=str, default=""]

# Plots
- Top left: data vs prediction
- Top Right: the graph of network (red connections mean negative weights, blue means positive, and opacity is based on the absolute value of the weights, so smaller weight, lower opacity)
- Bottom Left: Mean Squared Loss through epochs
- Bottom Right: Binary Cross Entropy Loss through epochs

to use you can clone this repository and install the packages needed in requirements.txt and run main.py
## $2e^{-x}(\sin(5x)+x\cos(5x))$
| Dense Configuration | Convolutional Configuration |
| :-------: | :-------: |
| ![densenet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/FCNN_7x7_1.gif) | ![convnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/CNN_7x7_1.gif) |
| ![denseresnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/DenseResNet_7x7_1.gif) | ![convresnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/ConvResNet_7x7_1.gif) |

| Custom Configuration |
| :------------------: |
| ![customnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/CustomNet_7x7_1.gif) |

## Observations
- FCNN and DenseResNet change more uniformly through epochs because of their dense configuration, while CNN, ConvResNet and CustomNet change very "noisily"
- Activation function Mish() works best for all configurations
- MSELoss(), L1Loss(), HuberLoss(), SmoothL1Loss(), which have similar formulas, work best for this problem
- FCNN, albeit its dense configuration, converges best throughout all different hyperparameters

## To-do
- add a heuristic approach for optimizing a Neural Network (CustomNet) by pruning connections e.g.: Grey Wolves Optimization, Genetic Algorith, etc.

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

Epoch      1 | MSELoss: 0.369860 | BCEWithLogitsLoss: 0.7857
Epoch     10 | MSELoss: 0.353828 | BCEWithLogitsLoss: 0.6948
Epoch     20 | MSELoss: 0.338907 | BCEWithLogitsLoss: 0.7004
Epoch     30 | MSELoss: 0.302386 | BCEWithLogitsLoss: 0.6966
Epoch     40 | MSELoss: 0.221528 | BCEWithLogitsLoss: 0.6235
Epoch     50 | MSELoss: 0.184860 | BCEWithLogitsLoss: 0.6225
Epoch     60 | MSELoss: 0.161507 | BCEWithLogitsLoss: 0.5908
Epoch     70 | MSELoss: 0.144004 | BCEWithLogitsLoss: 0.5583
Epoch     80 | MSELoss: 0.134603 | BCEWithLogitsLoss: 0.5242
Epoch     90 | MSELoss: 0.127729 | BCEWithLogitsLoss: 0.5361
Epoch    100 | MSELoss: 0.122855 | BCEWithLogitsLoss: 0.5340
Epoch    110 | MSELoss: 0.117852 | BCEWithLogitsLoss: 0.5244
Epoch    120 | MSELoss: 0.111541 | BCEWithLogitsLoss: 0.5272
Epoch    130 | MSELoss: 0.105780 | BCEWithLogitsLoss: 0.5343
Epoch    140 | MSELoss: 0.095358 | BCEWithLogitsLoss: 0.5268
Epoch    150 | MSELoss: 0.096383 | BCEWithLogitsLoss: 0.5468
Epoch    160 | MSELoss: 0.075440 | BCEWithLogitsLoss: 0.5078
Epoch    170 | MSELoss: 0.065446 | BCEWithLogitsLoss: 0.4874
Epoch    180 | MSELoss: 0.058088 | BCEWithLogitsLoss: 0.4668
Epoch    190 | MSELoss: 0.055341 | BCEWithLogitsLoss: 0.4734
Epoch    200 | MSELoss: 0.054157 | BCEWithLogitsLoss: 0.4612
Epoch    210 | MSELoss: 0.052775 | BCEWithLogitsLoss: 0.4637
Epoch    220 | MSELoss: 0.051528 | BCEWithLogitsLoss: 0.4637
Epoch    230 | MSELoss: 0.050384 | BCEWithLogitsLoss: 0.4626
Epoch    240 | MSELoss: 0.049111 | BCEWithLogitsLoss: 0.4615
Epoch    250 | MSELoss: 0.047710 | BCEWithLogitsLoss: 0.4599
Epoch    260 | MSELoss: 0.046156 | BCEWithLogitsLoss: 0.4590
Epoch    270 | MSELoss: 0.044377 | BCEWithLogitsLoss: 0.4579
Epoch    280 | MSELoss: 0.042231 | BCEWithLogitsLoss: 0.4559
Epoch    290 | MSELoss: 0.039540 | BCEWithLogitsLoss: 0.4553
Epoch    300 | MSELoss: 0.036089 | BCEWithLogitsLoss: 0.4528
Epoch    310 | MSELoss: 0.032937 | BCEWithLogitsLoss: 0.4512
Epoch    320 | MSELoss: 0.031084 | BCEWithLogitsLoss: 0.4472
Epoch    330 | MSELoss: 0.029863 | BCEWithLogitsLoss: 0.4451
Epoch    340 | MSELoss: 0.028996 | BCEWithLogitsLoss: 0.4450
Epoch    350 | MSELoss: 0.028312 | BCEWithLogitsLoss: 0.4439
Epoch    360 | MSELoss: 0.027678 | BCEWithLogitsLoss: 0.4441
Epoch    370 | MSELoss: 0.027173 | BCEWithLogitsLoss: 0.4432
Epoch    380 | MSELoss: 0.026717 | BCEWithLogitsLoss: 0.4424
Epoch    390 | MSELoss: 0.026344 | BCEWithLogitsLoss: 0.4422
Epoch    400 | MSELoss: 0.026126 | BCEWithLogitsLoss: 0.4422
Epoch    410 | MSELoss: 0.025702 | BCEWithLogitsLoss: 0.4427
Epoch    420 | MSELoss: 0.025329 | BCEWithLogitsLoss: 0.4405
Epoch    430 | MSELoss: 0.025025 | BCEWithLogitsLoss: 0.4389
Epoch    440 | MSELoss: 0.024971 | BCEWithLogitsLoss: 0.4362
Epoch    450 | MSELoss: 0.024528 | BCEWithLogitsLoss: 0.4367
Epoch    460 | MSELoss: 0.024143 | BCEWithLogitsLoss: 0.4422
Epoch    470 | MSELoss: 0.023884 | BCEWithLogitsLoss: 0.4376
Epoch    480 | MSELoss: 0.024191 | BCEWithLogitsLoss: 0.4312
Epoch    490 | MSELoss: 0.023276 | BCEWithLogitsLoss: 0.4421
Epoch    500 | MSELoss: 0.023179 | BCEWithLogitsLoss: 0.4325
Epoch    510 | MSELoss: 0.022745 | BCEWithLogitsLoss: 0.4329
Epoch    520 | MSELoss: 0.022631 | BCEWithLogitsLoss: 0.4308
Epoch    530 | MSELoss: 0.023111 | BCEWithLogitsLoss: 0.4507
Epoch    540 | MSELoss: 0.021880 | BCEWithLogitsLoss: 0.4465
Epoch    550 | MSELoss: 0.020925 | BCEWithLogitsLoss: 0.4381
Epoch    560 | MSELoss: 0.021498 | BCEWithLogitsLoss: 0.4478
Epoch    570 | MSELoss: 0.020338 | BCEWithLogitsLoss: 0.4323
Epoch    580 | MSELoss: 0.020059 | BCEWithLogitsLoss: 0.4441
Epoch    590 | MSELoss: 0.020002 | BCEWithLogitsLoss: 0.4463
Epoch    600 | MSELoss: 0.018941 | BCEWithLogitsLoss: 0.4416
Epoch    610 | MSELoss: 0.018217 | BCEWithLogitsLoss: 0.4373
Epoch    620 | MSELoss: 0.037882 | BCEWithLogitsLoss: 0.3941
Epoch    630 | MSELoss: 0.023819 | BCEWithLogitsLoss: 0.4615
Epoch    640 | MSELoss: 0.018471 | BCEWithLogitsLoss: 0.4402
Epoch    650 | MSELoss: 0.017485 | BCEWithLogitsLoss: 0.4333
Epoch    660 | MSELoss: 0.016863 | BCEWithLogitsLoss: 0.4358
Epoch    670 | MSELoss: 0.016088 | BCEWithLogitsLoss: 0.4348
Epoch    680 | MSELoss: 0.015583 | BCEWithLogitsLoss: 0.4329
Epoch    690 | MSELoss: 0.015015 | BCEWithLogitsLoss: 0.4328
Epoch    700 | MSELoss: 0.014479 | BCEWithLogitsLoss: 0.4328
Epoch    710 | MSELoss: 0.014642 | BCEWithLogitsLoss: 0.4401
Epoch    720 | MSELoss: 0.024191 | BCEWithLogitsLoss: 0.4705
Epoch    730 | MSELoss: 0.016206 | BCEWithLogitsLoss: 0.4504
Epoch    740 | MSELoss: 0.015465 | BCEWithLogitsLoss: 0.4454
Epoch    750 | MSELoss: 0.013525 | BCEWithLogitsLoss: 0.4363
Epoch    760 | MSELoss: 0.012973 | BCEWithLogitsLoss: 0.4321
Epoch    770 | MSELoss: 0.012600 | BCEWithLogitsLoss: 0.4329
Epoch    780 | MSELoss: 0.012175 | BCEWithLogitsLoss: 0.4314
Epoch    790 | MSELoss: 0.011806 | BCEWithLogitsLoss: 0.4291
Epoch    800 | MSELoss: 0.011472 | BCEWithLogitsLoss: 0.4288
Epoch    810 | MSELoss: 0.011172 | BCEWithLogitsLoss: 0.4285
Epoch    820 | MSELoss: 0.010909 | BCEWithLogitsLoss: 0.4288
Epoch    830 | MSELoss: 0.010703 | BCEWithLogitsLoss: 0.4271
Epoch    840 | MSELoss: 0.017507 | BCEWithLogitsLoss: 0.4028
Epoch    850 | MSELoss: 0.010885 | BCEWithLogitsLoss: 0.4289
Epoch    860 | MSELoss: 0.010943 | BCEWithLogitsLoss: 0.4269
Epoch    870 | MSELoss: 0.011491 | BCEWithLogitsLoss: 0.4203
Epoch    880 | MSELoss: 0.010931 | BCEWithLogitsLoss: 0.4232
Epoch    890 | MSELoss: 0.010518 | BCEWithLogitsLoss: 0.4255
Epoch    900 | MSELoss: 0.010352 | BCEWithLogitsLoss: 0.4262
Epoch    910 | MSELoss: 0.010247 | BCEWithLogitsLoss: 0.4262
Epoch    920 | MSELoss: 0.010161 | BCEWithLogitsLoss: 0.4269
Epoch    930 | MSELoss: 0.010107 | BCEWithLogitsLoss: 0.4277
Epoch    940 | MSELoss: 0.010063 | BCEWithLogitsLoss: 0.4270
Epoch    950 | MSELoss: 0.010029 | BCEWithLogitsLoss: 0.4269
Epoch    960 | MSELoss: 0.010000 | BCEWithLogitsLoss: 0.4269
Epoch    970 | MSELoss: 0.009975 | BCEWithLogitsLoss: 0.4268
Epoch    980 | MSELoss: 0.009953 | BCEWithLogitsLoss: 0.4268
Epoch    990 | MSELoss: 0.009932 | BCEWithLogitsLoss: 0.4268
Epoch   1000 | MSELoss: 0.009913 | BCEWithLogitsLoss: 0.4267

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

Epoch      1 | MSELoss: 0.558304 | BCEWithLogitsLoss: 0.5616
Epoch     10 | MSELoss: 0.356604 | BCEWithLogitsLoss: 0.7486
Epoch     20 | MSELoss: 0.346754 | BCEWithLogitsLoss: 0.7392
Epoch     30 | MSELoss: 0.350115 | BCEWithLogitsLoss: 0.7200
Epoch     40 | MSELoss: 0.341238 | BCEWithLogitsLoss: 0.7064
Epoch     50 | MSELoss: 0.327231 | BCEWithLogitsLoss: 0.7053
Epoch     60 | MSELoss: 0.288273 | BCEWithLogitsLoss: 0.6750
Epoch     70 | MSELoss: 0.273206 | BCEWithLogitsLoss: 0.6443
Epoch     80 | MSELoss: 0.244539 | BCEWithLogitsLoss: 0.6375
Epoch     90 | MSELoss: 0.226977 | BCEWithLogitsLoss: 0.6115
Epoch    100 | MSELoss: 0.194276 | BCEWithLogitsLoss: 0.5931
Epoch    110 | MSELoss: 0.156088 | BCEWithLogitsLoss: 0.5793
Epoch    120 | MSELoss: 0.176363 | BCEWithLogitsLoss: 0.5606
Epoch    130 | MSELoss: 0.162090 | BCEWithLogitsLoss: 0.5582
Epoch    140 | MSELoss: 0.140848 | BCEWithLogitsLoss: 0.5758
Epoch    150 | MSELoss: 0.133529 | BCEWithLogitsLoss: 0.5466
Epoch    160 | MSELoss: 0.128973 | BCEWithLogitsLoss: 0.5205
Epoch    170 | MSELoss: 0.129945 | BCEWithLogitsLoss: 0.5542
Epoch    180 | MSELoss: 0.118875 | BCEWithLogitsLoss: 0.5160
Epoch    190 | MSELoss: 0.121775 | BCEWithLogitsLoss: 0.5468
Epoch    200 | MSELoss: 0.118642 | BCEWithLogitsLoss: 0.5382
Epoch    210 | MSELoss: 0.100096 | BCEWithLogitsLoss: 0.5045
Epoch    220 | MSELoss: 0.096644 | BCEWithLogitsLoss: 0.5243
Epoch    230 | MSELoss: 0.096238 | BCEWithLogitsLoss: 0.5086
Epoch    240 | MSELoss: 0.095157 | BCEWithLogitsLoss: 0.4721
Epoch    250 | MSELoss: 0.089068 | BCEWithLogitsLoss: 0.4921
Epoch    260 | MSELoss: 0.081565 | BCEWithLogitsLoss: 0.5084
Epoch    270 | MSELoss: 0.087838 | BCEWithLogitsLoss: 0.4786
Epoch    280 | MSELoss: 0.073979 | BCEWithLogitsLoss: 0.4823
Epoch    290 | MSELoss: 0.078330 | BCEWithLogitsLoss: 0.4750
Epoch    300 | MSELoss: 0.076967 | BCEWithLogitsLoss: 0.5045
Epoch    310 | MSELoss: 0.076341 | BCEWithLogitsLoss: 0.4828
Epoch    320 | MSELoss: 0.069975 | BCEWithLogitsLoss: 0.4713
Epoch    330 | MSELoss: 0.075397 | BCEWithLogitsLoss: 0.4860
Epoch    340 | MSELoss: 0.074844 | BCEWithLogitsLoss: 0.4848
Epoch    350 | MSELoss: 0.069503 | BCEWithLogitsLoss: 0.4825
Epoch    360 | MSELoss: 0.069232 | BCEWithLogitsLoss: 0.4756
Epoch    370 | MSELoss: 0.067777 | BCEWithLogitsLoss: 0.4750
Epoch    380 | MSELoss: 0.068364 | BCEWithLogitsLoss: 0.4791
Epoch    390 | MSELoss: 0.066212 | BCEWithLogitsLoss: 0.4757
Epoch    400 | MSELoss: 0.066685 | BCEWithLogitsLoss: 0.4764
Epoch    410 | MSELoss: 0.064080 | BCEWithLogitsLoss: 0.4798
Epoch    420 | MSELoss: 0.061066 | BCEWithLogitsLoss: 0.4795
Epoch    430 | MSELoss: 0.062162 | BCEWithLogitsLoss: 0.4799
Epoch    440 | MSELoss: 0.063348 | BCEWithLogitsLoss: 0.4619
Epoch    450 | MSELoss: 0.062242 | BCEWithLogitsLoss: 0.4617
Epoch    460 | MSELoss: 0.057982 | BCEWithLogitsLoss: 0.4699
Epoch    470 | MSELoss: 0.065270 | BCEWithLogitsLoss: 0.4591
Epoch    480 | MSELoss: 0.054042 | BCEWithLogitsLoss: 0.4591
Epoch    490 | MSELoss: 0.046970 | BCEWithLogitsLoss: 0.4614
Epoch    500 | MSELoss: 0.043963 | BCEWithLogitsLoss: 0.4588
Epoch    510 | MSELoss: 0.044381 | BCEWithLogitsLoss: 0.4460
Epoch    520 | MSELoss: 0.040196 | BCEWithLogitsLoss: 0.4439
Epoch    530 | MSELoss: 0.034156 | BCEWithLogitsLoss: 0.4515
Epoch    540 | MSELoss: 0.034546 | BCEWithLogitsLoss: 0.4434
Epoch    550 | MSELoss: 0.024743 | BCEWithLogitsLoss: 0.4443
Epoch    560 | MSELoss: 0.024979 | BCEWithLogitsLoss: 0.4447
Epoch    570 | MSELoss: 0.021222 | BCEWithLogitsLoss: 0.4477
Epoch    580 | MSELoss: 0.019685 | BCEWithLogitsLoss: 0.4393
Epoch    590 | MSELoss: 0.027640 | BCEWithLogitsLoss: 0.4386
Epoch    600 | MSELoss: 0.026931 | BCEWithLogitsLoss: 0.4397
Epoch    610 | MSELoss: 0.018922 | BCEWithLogitsLoss: 0.4244
Epoch    620 | MSELoss: 0.019290 | BCEWithLogitsLoss: 0.4404
Epoch    630 | MSELoss: 0.023039 | BCEWithLogitsLoss: 0.4380
Epoch    640 | MSELoss: 0.020921 | BCEWithLogitsLoss: 0.4446
Epoch    650 | MSELoss: 0.019314 | BCEWithLogitsLoss: 0.4373
Epoch    660 | MSELoss: 0.017931 | BCEWithLogitsLoss: 0.4404
Epoch    670 | MSELoss: 0.020652 | BCEWithLogitsLoss: 0.4394
Epoch    680 | MSELoss: 0.016887 | BCEWithLogitsLoss: 0.4326
Epoch    690 | MSELoss: 0.017134 | BCEWithLogitsLoss: 0.4223
Epoch    700 | MSELoss: 0.016101 | BCEWithLogitsLoss: 0.4282
Epoch    710 | MSELoss: 0.016068 | BCEWithLogitsLoss: 0.4366
Epoch    720 | MSELoss: 0.018127 | BCEWithLogitsLoss: 0.4428
Epoch    730 | MSELoss: 0.016903 | BCEWithLogitsLoss: 0.4311
Epoch    740 | MSELoss: 0.019188 | BCEWithLogitsLoss: 0.4379
Epoch    750 | MSELoss: 0.015825 | BCEWithLogitsLoss: 0.4408
Epoch    760 | MSELoss: 0.014034 | BCEWithLogitsLoss: 0.4341
Epoch    770 | MSELoss: 0.014941 | BCEWithLogitsLoss: 0.4349
Epoch    780 | MSELoss: 0.016786 | BCEWithLogitsLoss: 0.4333
Epoch    790 | MSELoss: 0.015210 | BCEWithLogitsLoss: 0.4389
Epoch    800 | MSELoss: 0.015828 | BCEWithLogitsLoss: 0.4383
Epoch    810 | MSELoss: 0.014353 | BCEWithLogitsLoss: 0.4414
Epoch    820 | MSELoss: 0.018014 | BCEWithLogitsLoss: 0.4362
Epoch    830 | MSELoss: 0.016642 | BCEWithLogitsLoss: 0.4382
Epoch    840 | MSELoss: 0.017562 | BCEWithLogitsLoss: 0.4456
Epoch    850 | MSELoss: 0.014939 | BCEWithLogitsLoss: 0.4349
Epoch    860 | MSELoss: 0.019742 | BCEWithLogitsLoss: 0.4402
Epoch    870 | MSELoss: 0.017761 | BCEWithLogitsLoss: 0.4482
Epoch    880 | MSELoss: 0.016106 | BCEWithLogitsLoss: 0.4381
Epoch    890 | MSELoss: 0.015062 | BCEWithLogitsLoss: 0.4273
Epoch    900 | MSELoss: 0.014891 | BCEWithLogitsLoss: 0.4336
Epoch    910 | MSELoss: 0.012692 | BCEWithLogitsLoss: 0.4327
Epoch    920 | MSELoss: 0.015562 | BCEWithLogitsLoss: 0.4271
Epoch    930 | MSELoss: 0.015143 | BCEWithLogitsLoss: 0.4361
Epoch    940 | MSELoss: 0.013268 | BCEWithLogitsLoss: 0.4289
Epoch    950 | MSELoss: 0.015347 | BCEWithLogitsLoss: 0.4373
Epoch    960 | MSELoss: 0.015476 | BCEWithLogitsLoss: 0.4338
Epoch    970 | MSELoss: 0.014084 | BCEWithLogitsLoss: 0.4367
Epoch    980 | MSELoss: 0.012359 | BCEWithLogitsLoss: 0.4296
Epoch    990 | MSELoss: 0.015192 | BCEWithLogitsLoss: 0.4363
Epoch   1000 | MSELoss: 0.015554 | BCEWithLogitsLoss: 0.4262

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

Epoch      1 | MSELoss: 0.538941 | BCEWithLogitsLoss: 0.9089
Epoch     10 | MSELoss: 0.351394 | BCEWithLogitsLoss: 0.7677
Epoch     20 | MSELoss: 0.325336 | BCEWithLogitsLoss: 0.6766
Epoch     30 | MSELoss: 0.308420 | BCEWithLogitsLoss: 0.7021
Epoch     40 | MSELoss: 0.289779 | BCEWithLogitsLoss: 0.6845
Epoch     50 | MSELoss: 0.259498 | BCEWithLogitsLoss: 0.6461
Epoch     60 | MSELoss: 0.241324 | BCEWithLogitsLoss: 0.6242
Epoch     70 | MSELoss: 0.220212 | BCEWithLogitsLoss: 0.6223
Epoch     80 | MSELoss: 0.199045 | BCEWithLogitsLoss: 0.6139
Epoch     90 | MSELoss: 0.175103 | BCEWithLogitsLoss: 0.5881
Epoch    100 | MSELoss: 0.158508 | BCEWithLogitsLoss: 0.5643
Epoch    110 | MSELoss: 0.151405 | BCEWithLogitsLoss: 0.5605
Epoch    120 | MSELoss: 0.146460 | BCEWithLogitsLoss: 0.5582
Epoch    130 | MSELoss: 0.141994 | BCEWithLogitsLoss: 0.5553
Epoch    140 | MSELoss: 0.137419 | BCEWithLogitsLoss: 0.5517
Epoch    150 | MSELoss: 0.135441 | BCEWithLogitsLoss: 0.5261
Epoch    160 | MSELoss: 0.129631 | BCEWithLogitsLoss: 0.5274
Epoch    170 | MSELoss: 0.122669 | BCEWithLogitsLoss: 0.5331
Epoch    180 | MSELoss: 0.114422 | BCEWithLogitsLoss: 0.5299
Epoch    190 | MSELoss: 0.106912 | BCEWithLogitsLoss: 0.5512
Epoch    200 | MSELoss: 0.089398 | BCEWithLogitsLoss: 0.5250
Epoch    210 | MSELoss: 0.075572 | BCEWithLogitsLoss: 0.4680
Epoch    220 | MSELoss: 0.055868 | BCEWithLogitsLoss: 0.4995
Epoch    230 | MSELoss: 0.041425 | BCEWithLogitsLoss: 0.4584
Epoch    240 | MSELoss: 0.038010 | BCEWithLogitsLoss: 0.4373
Epoch    250 | MSELoss: 0.034867 | BCEWithLogitsLoss: 0.4423
Epoch    260 | MSELoss: 0.032872 | BCEWithLogitsLoss: 0.4466
Epoch    270 | MSELoss: 0.031669 | BCEWithLogitsLoss: 0.4527
Epoch    280 | MSELoss: 0.030922 | BCEWithLogitsLoss: 0.4512
Epoch    290 | MSELoss: 0.030379 | BCEWithLogitsLoss: 0.4468
Epoch    300 | MSELoss: 0.030263 | BCEWithLogitsLoss: 0.4550
Epoch    310 | MSELoss: 0.029492 | BCEWithLogitsLoss: 0.4447
Epoch    320 | MSELoss: 0.029106 | BCEWithLogitsLoss: 0.4458
Epoch    330 | MSELoss: 0.029062 | BCEWithLogitsLoss: 0.4373
Epoch    340 | MSELoss: 0.028327 | BCEWithLogitsLoss: 0.4408
Epoch    350 | MSELoss: 0.027862 | BCEWithLogitsLoss: 0.4419
Epoch    360 | MSELoss: 0.027483 | BCEWithLogitsLoss: 0.4445
Epoch    370 | MSELoss: 0.027507 | BCEWithLogitsLoss: 0.4524
Epoch    380 | MSELoss: 0.027436 | BCEWithLogitsLoss: 0.4550
Epoch    390 | MSELoss: 0.026379 | BCEWithLogitsLoss: 0.4451
Epoch    400 | MSELoss: 0.025970 | BCEWithLogitsLoss: 0.4400
Epoch    410 | MSELoss: 0.036026 | BCEWithLogitsLoss: 0.3989
Epoch    420 | MSELoss: 0.025815 | BCEWithLogitsLoss: 0.4307
Epoch    430 | MSELoss: 0.025791 | BCEWithLogitsLoss: 0.4556
Epoch    440 | MSELoss: 0.024400 | BCEWithLogitsLoss: 0.4420
Epoch    450 | MSELoss: 0.024000 | BCEWithLogitsLoss: 0.4367
Epoch    460 | MSELoss: 0.023469 | BCEWithLogitsLoss: 0.4372
Epoch    470 | MSELoss: 0.022935 | BCEWithLogitsLoss: 0.4385
Epoch    480 | MSELoss: 0.022415 | BCEWithLogitsLoss: 0.4409
Epoch    490 | MSELoss: 0.021901 | BCEWithLogitsLoss: 0.4376
Epoch    500 | MSELoss: 0.022110 | BCEWithLogitsLoss: 0.4271
Epoch    510 | MSELoss: 0.025909 | BCEWithLogitsLoss: 0.4696
Epoch    520 | MSELoss: 0.022276 | BCEWithLogitsLoss: 0.4562
Epoch    530 | MSELoss: 0.020796 | BCEWithLogitsLoss: 0.4501
Epoch    540 | MSELoss: 0.019750 | BCEWithLogitsLoss: 0.4431
Epoch    550 | MSELoss: 0.019136 | BCEWithLogitsLoss: 0.4339
Epoch    560 | MSELoss: 0.018617 | BCEWithLogitsLoss: 0.4389
Epoch    570 | MSELoss: 0.018241 | BCEWithLogitsLoss: 0.4312
Epoch    580 | MSELoss: 0.017672 | BCEWithLogitsLoss: 0.4356
Epoch    590 | MSELoss: 0.022460 | BCEWithLogitsLoss: 0.4673
Epoch    600 | MSELoss: 0.019115 | BCEWithLogitsLoss: 0.4571
Epoch    610 | MSELoss: 0.017067 | BCEWithLogitsLoss: 0.4246
Epoch    620 | MSELoss: 0.016558 | BCEWithLogitsLoss: 0.4274
Epoch    630 | MSELoss: 0.015781 | BCEWithLogitsLoss: 0.4309
Epoch    640 | MSELoss: 0.015348 | BCEWithLogitsLoss: 0.4311
Epoch    650 | MSELoss: 0.014896 | BCEWithLogitsLoss: 0.4324
Epoch    660 | MSELoss: 0.014482 | BCEWithLogitsLoss: 0.4340
Epoch    670 | MSELoss: 0.014395 | BCEWithLogitsLoss: 0.4399
Epoch    680 | MSELoss: 0.015194 | BCEWithLogitsLoss: 0.4499
Epoch    690 | MSELoss: 0.014188 | BCEWithLogitsLoss: 0.4445
Epoch    700 | MSELoss: 0.014277 | BCEWithLogitsLoss: 0.4176
Epoch    710 | MSELoss: 0.013443 | BCEWithLogitsLoss: 0.4415
Epoch    720 | MSELoss: 0.012809 | BCEWithLogitsLoss: 0.4250
Epoch    730 | MSELoss: 0.018153 | BCEWithLogitsLoss: 0.3994
Epoch    740 | MSELoss: 0.015259 | BCEWithLogitsLoss: 0.4562
Epoch    750 | MSELoss: 0.011976 | BCEWithLogitsLoss: 0.4313
Epoch    760 | MSELoss: 0.012246 | BCEWithLogitsLoss: 0.4208
Epoch    770 | MSELoss: 0.011875 | BCEWithLogitsLoss: 0.4367
Epoch    780 | MSELoss: 0.011523 | BCEWithLogitsLoss: 0.4259
Epoch    790 | MSELoss: 0.012917 | BCEWithLogitsLoss: 0.4127
Epoch    800 | MSELoss: 0.011292 | BCEWithLogitsLoss: 0.4265
Epoch    810 | MSELoss: 0.012819 | BCEWithLogitsLoss: 0.4113
Epoch    820 | MSELoss: 0.011063 | BCEWithLogitsLoss: 0.4308
Epoch    830 | MSELoss: 0.011169 | BCEWithLogitsLoss: 0.4352
Epoch    840 | MSELoss: 0.011012 | BCEWithLogitsLoss: 0.4239
Epoch    850 | MSELoss: 0.010877 | BCEWithLogitsLoss: 0.4316
Epoch    860 | MSELoss: 0.010813 | BCEWithLogitsLoss: 0.4311
Epoch    870 | MSELoss: 0.011322 | BCEWithLogitsLoss: 0.4386
Epoch    880 | MSELoss: 0.011896 | BCEWithLogitsLoss: 0.4444
Epoch    890 | MSELoss: 0.011489 | BCEWithLogitsLoss: 0.4408
Epoch    900 | MSELoss: 0.011104 | BCEWithLogitsLoss: 0.4192
Epoch    910 | MSELoss: 0.010678 | BCEWithLogitsLoss: 0.4246
Epoch    920 | MSELoss: 0.010688 | BCEWithLogitsLoss: 0.4325
Epoch    930 | MSELoss: 0.010585 | BCEWithLogitsLoss: 0.4253
Epoch    940 | MSELoss: 0.010542 | BCEWithLogitsLoss: 0.4297
Epoch    950 | MSELoss: 0.010502 | BCEWithLogitsLoss: 0.4268
Epoch    960 | MSELoss: 0.010502 | BCEWithLogitsLoss: 0.4254
Epoch    970 | MSELoss: 0.011081 | BCEWithLogitsLoss: 0.4170
Epoch    980 | MSELoss: 0.015411 | BCEWithLogitsLoss: 0.3977
Epoch    990 | MSELoss: 0.010887 | BCEWithLogitsLoss: 0.4182
Epoch   1000 | MSELoss: 0.010705 | BCEWithLogitsLoss: 0.4351

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

Epoch      1 | MSELoss: 0.623714 | BCEWithLogitsLoss: 0.5426
Epoch     10 | MSELoss: 0.367148 | BCEWithLogitsLoss: 0.6648
Epoch     20 | MSELoss: 0.347607 | BCEWithLogitsLoss: 0.7191
Epoch     30 | MSELoss: 0.335924 | BCEWithLogitsLoss: 0.7258
Epoch     40 | MSELoss: 0.322583 | BCEWithLogitsLoss: 0.6934
Epoch     50 | MSELoss: 0.313355 | BCEWithLogitsLoss: 0.7036
Epoch     60 | MSELoss: 0.296099 | BCEWithLogitsLoss: 0.6905
Epoch     70 | MSELoss: 0.291025 | BCEWithLogitsLoss: 0.6701
Epoch     80 | MSELoss: 0.266367 | BCEWithLogitsLoss: 0.6548
Epoch     90 | MSELoss: 0.266693 | BCEWithLogitsLoss: 0.6558
Epoch    100 | MSELoss: 0.269988 | BCEWithLogitsLoss: 0.6594
Epoch    110 | MSELoss: 0.248187 | BCEWithLogitsLoss: 0.6333
Epoch    120 | MSELoss: 0.232191 | BCEWithLogitsLoss: 0.6174
Epoch    130 | MSELoss: 0.201907 | BCEWithLogitsLoss: 0.5926
Epoch    140 | MSELoss: 0.173376 | BCEWithLogitsLoss: 0.5669
Epoch    150 | MSELoss: 0.177314 | BCEWithLogitsLoss: 0.5577
Epoch    160 | MSELoss: 0.158160 | BCEWithLogitsLoss: 0.5762
Epoch    170 | MSELoss: 0.149094 | BCEWithLogitsLoss: 0.5531
Epoch    180 | MSELoss: 0.139312 | BCEWithLogitsLoss: 0.5494
Epoch    190 | MSELoss: 0.144594 | BCEWithLogitsLoss: 0.5353
Epoch    200 | MSELoss: 0.132330 | BCEWithLogitsLoss: 0.5440
Epoch    210 | MSELoss: 0.123436 | BCEWithLogitsLoss: 0.5422
Epoch    220 | MSELoss: 0.115215 | BCEWithLogitsLoss: 0.5271
Epoch    230 | MSELoss: 0.113099 | BCEWithLogitsLoss: 0.5217
Epoch    240 | MSELoss: 0.113292 | BCEWithLogitsLoss: 0.5190
Epoch    250 | MSELoss: 0.107586 | BCEWithLogitsLoss: 0.5174
Epoch    260 | MSELoss: 0.108346 | BCEWithLogitsLoss: 0.5254
Epoch    270 | MSELoss: 0.121461 | BCEWithLogitsLoss: 0.5169
Epoch    280 | MSELoss: 0.104604 | BCEWithLogitsLoss: 0.4957
Epoch    290 | MSELoss: 0.097896 | BCEWithLogitsLoss: 0.4898
Epoch    300 | MSELoss: 0.081946 | BCEWithLogitsLoss: 0.5181
Epoch    310 | MSELoss: 0.089685 | BCEWithLogitsLoss: 0.5162
Epoch    320 | MSELoss: 0.085901 | BCEWithLogitsLoss: 0.4741
Epoch    330 | MSELoss: 0.077679 | BCEWithLogitsLoss: 0.4838
Epoch    340 | MSELoss: 0.089636 | BCEWithLogitsLoss: 0.5118
Epoch    350 | MSELoss: 0.072833 | BCEWithLogitsLoss: 0.4684
Epoch    360 | MSELoss: 0.071407 | BCEWithLogitsLoss: 0.4699
Epoch    370 | MSELoss: 0.070698 | BCEWithLogitsLoss: 0.4744
Epoch    380 | MSELoss: 0.074596 | BCEWithLogitsLoss: 0.5064
Epoch    390 | MSELoss: 0.069045 | BCEWithLogitsLoss: 0.4802
Epoch    400 | MSELoss: 0.072215 | BCEWithLogitsLoss: 0.4768
Epoch    410 | MSELoss: 0.075737 | BCEWithLogitsLoss: 0.4789
Epoch    420 | MSELoss: 0.080706 | BCEWithLogitsLoss: 0.4788
Epoch    430 | MSELoss: 0.067037 | BCEWithLogitsLoss: 0.4870
Epoch    440 | MSELoss: 0.066008 | BCEWithLogitsLoss: 0.4615
Epoch    450 | MSELoss: 0.067466 | BCEWithLogitsLoss: 0.4774
Epoch    460 | MSELoss: 0.066599 | BCEWithLogitsLoss: 0.4912
Epoch    470 | MSELoss: 0.059180 | BCEWithLogitsLoss: 0.4706
Epoch    480 | MSELoss: 0.061668 | BCEWithLogitsLoss: 0.4788
Epoch    490 | MSELoss: 0.051118 | BCEWithLogitsLoss: 0.4608
Epoch    500 | MSELoss: 0.063831 | BCEWithLogitsLoss: 0.4820
Epoch    510 | MSELoss: 0.044564 | BCEWithLogitsLoss: 0.4603
Epoch    520 | MSELoss: 0.048822 | BCEWithLogitsLoss: 0.4477
Epoch    530 | MSELoss: 0.055216 | BCEWithLogitsLoss: 0.4620
Epoch    540 | MSELoss: 0.038652 | BCEWithLogitsLoss: 0.4514
Epoch    550 | MSELoss: 0.036977 | BCEWithLogitsLoss: 0.4430
Epoch    560 | MSELoss: 0.035362 | BCEWithLogitsLoss: 0.4678
Epoch    570 | MSELoss: 0.034565 | BCEWithLogitsLoss: 0.4494
Epoch    580 | MSELoss: 0.042006 | BCEWithLogitsLoss: 0.4595
Epoch    590 | MSELoss: 0.029948 | BCEWithLogitsLoss: 0.4546
Epoch    600 | MSELoss: 0.027912 | BCEWithLogitsLoss: 0.4266
Epoch    610 | MSELoss: 0.029105 | BCEWithLogitsLoss: 0.4475
Epoch    620 | MSELoss: 0.032913 | BCEWithLogitsLoss: 0.4373
Epoch    630 | MSELoss: 0.019348 | BCEWithLogitsLoss: 0.4475
Epoch    640 | MSELoss: 0.022484 | BCEWithLogitsLoss: 0.4294
Epoch    650 | MSELoss: 0.022204 | BCEWithLogitsLoss: 0.4330
Epoch    660 | MSELoss: 0.023199 | BCEWithLogitsLoss: 0.4261
Epoch    670 | MSELoss: 0.028390 | BCEWithLogitsLoss: 0.4518
Epoch    680 | MSELoss: 0.023184 | BCEWithLogitsLoss: 0.4384
Epoch    690 | MSELoss: 0.022215 | BCEWithLogitsLoss: 0.4335
Epoch    700 | MSELoss: 0.024064 | BCEWithLogitsLoss: 0.4306
Epoch    710 | MSELoss: 0.024303 | BCEWithLogitsLoss: 0.4444
Epoch    720 | MSELoss: 0.020371 | BCEWithLogitsLoss: 0.4395
Epoch    730 | MSELoss: 0.023594 | BCEWithLogitsLoss: 0.4410
Epoch    740 | MSELoss: 0.019205 | BCEWithLogitsLoss: 0.4223
Epoch    750 | MSELoss: 0.017483 | BCEWithLogitsLoss: 0.4243
Epoch    760 | MSELoss: 0.018061 | BCEWithLogitsLoss: 0.4519
Epoch    770 | MSELoss: 0.018261 | BCEWithLogitsLoss: 0.4258
Epoch    780 | MSELoss: 0.018851 | BCEWithLogitsLoss: 0.4298
Epoch    790 | MSELoss: 0.028290 | BCEWithLogitsLoss: 0.4369
Epoch    800 | MSELoss: 0.015693 | BCEWithLogitsLoss: 0.4376
Epoch    810 | MSELoss: 0.013977 | BCEWithLogitsLoss: 0.4317
Epoch    820 | MSELoss: 0.016071 | BCEWithLogitsLoss: 0.4422
Epoch    830 | MSELoss: 0.021041 | BCEWithLogitsLoss: 0.4293
Epoch    840 | MSELoss: 0.020218 | BCEWithLogitsLoss: 0.4454
Epoch    850 | MSELoss: 0.023633 | BCEWithLogitsLoss: 0.4474
Epoch    860 | MSELoss: 0.014538 | BCEWithLogitsLoss: 0.4288
Epoch    870 | MSELoss: 0.019081 | BCEWithLogitsLoss: 0.4303
Epoch    880 | MSELoss: 0.017297 | BCEWithLogitsLoss: 0.4416
Epoch    890 | MSELoss: 0.015614 | BCEWithLogitsLoss: 0.4228
Epoch    900 | MSELoss: 0.015670 | BCEWithLogitsLoss: 0.4294
Epoch    910 | MSELoss: 0.014951 | BCEWithLogitsLoss: 0.4243
Epoch    920 | MSELoss: 0.015530 | BCEWithLogitsLoss: 0.4322
Epoch    930 | MSELoss: 0.015707 | BCEWithLogitsLoss: 0.4397
Epoch    940 | MSELoss: 0.018289 | BCEWithLogitsLoss: 0.4399
Epoch    950 | MSELoss: 0.017864 | BCEWithLogitsLoss: 0.4429
Epoch    960 | MSELoss: 0.015163 | BCEWithLogitsLoss: 0.4406
Epoch    970 | MSELoss: 0.013762 | BCEWithLogitsLoss: 0.4297
Epoch    980 | MSELoss: 0.013639 | BCEWithLogitsLoss: 0.4419
Epoch    990 | MSELoss: 0.020468 | BCEWithLogitsLoss: 0.4288
Epoch   1000 | MSELoss: 0.014485 | BCEWithLogitsLoss: 0.4325

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

Epoch      1 | MSELoss: 0.365378 | BCEWithLogitsLoss: 0.7866
Epoch     10 | MSELoss: 0.341137 | BCEWithLogitsLoss: 0.6972
Epoch     20 | MSELoss: 0.332424 | BCEWithLogitsLoss: 0.7240
Epoch     30 | MSELoss: 0.321666 | BCEWithLogitsLoss: 0.7036
Epoch     40 | MSELoss: 0.308276 | BCEWithLogitsLoss: 0.6989
Epoch     50 | MSELoss: 0.293866 | BCEWithLogitsLoss: 0.6903
Epoch     60 | MSELoss: 0.281790 | BCEWithLogitsLoss: 0.6720
Epoch     70 | MSELoss: 0.272070 | BCEWithLogitsLoss: 0.6601
Epoch     80 | MSELoss: 0.265192 | BCEWithLogitsLoss: 0.6536
Epoch     90 | MSELoss: 0.259181 | BCEWithLogitsLoss: 0.6447
Epoch    100 | MSELoss: 0.246888 | BCEWithLogitsLoss: 0.6343
Epoch    110 | MSELoss: 0.220379 | BCEWithLogitsLoss: 0.6189
Epoch    120 | MSELoss: 0.181349 | BCEWithLogitsLoss: 0.5845
Epoch    130 | MSELoss: 0.152542 | BCEWithLogitsLoss: 0.5504
Epoch    140 | MSELoss: 0.133838 | BCEWithLogitsLoss: 0.5406
Epoch    150 | MSELoss: 0.111325 | BCEWithLogitsLoss: 0.5187
Epoch    160 | MSELoss: 0.090172 | BCEWithLogitsLoss: 0.5023
Epoch    170 | MSELoss: 0.072186 | BCEWithLogitsLoss: 0.4845
Epoch    180 | MSELoss: 0.062981 | BCEWithLogitsLoss: 0.4814
Epoch    190 | MSELoss: 0.058599 | BCEWithLogitsLoss: 0.4731
Epoch    200 | MSELoss: 0.055162 | BCEWithLogitsLoss: 0.4739
Epoch    210 | MSELoss: 0.051941 | BCEWithLogitsLoss: 0.4610
Epoch    220 | MSELoss: 0.048968 | BCEWithLogitsLoss: 0.4693
Epoch    230 | MSELoss: 0.046196 | BCEWithLogitsLoss: 0.4561
Epoch    240 | MSELoss: 0.043666 | BCEWithLogitsLoss: 0.4564
Epoch    250 | MSELoss: 0.041528 | BCEWithLogitsLoss: 0.4576
Epoch    260 | MSELoss: 0.039882 | BCEWithLogitsLoss: 0.4491
Epoch    270 | MSELoss: 0.038593 | BCEWithLogitsLoss: 0.4464
Epoch    280 | MSELoss: 0.037292 | BCEWithLogitsLoss: 0.4507
Epoch    290 | MSELoss: 0.036156 | BCEWithLogitsLoss: 0.4512
Epoch    300 | MSELoss: 0.035101 | BCEWithLogitsLoss: 0.4512
Epoch    310 | MSELoss: 0.035028 | BCEWithLogitsLoss: 0.4337
Epoch    320 | MSELoss: 0.033441 | BCEWithLogitsLoss: 0.4528
Epoch    330 | MSELoss: 0.032567 | BCEWithLogitsLoss: 0.4431
Epoch    340 | MSELoss: 0.031724 | BCEWithLogitsLoss: 0.4464
Epoch    350 | MSELoss: 0.030979 | BCEWithLogitsLoss: 0.4478
Epoch    360 | MSELoss: 0.030205 | BCEWithLogitsLoss: 0.4453
Epoch    370 | MSELoss: 0.030650 | BCEWithLogitsLoss: 0.4596
Epoch    380 | MSELoss: 0.028999 | BCEWithLogitsLoss: 0.4460
Epoch    390 | MSELoss: 0.028618 | BCEWithLogitsLoss: 0.4377
Epoch    400 | MSELoss: 0.028111 | BCEWithLogitsLoss: 0.4391
Epoch    410 | MSELoss: 0.027678 | BCEWithLogitsLoss: 0.4402
Epoch    420 | MSELoss: 0.027303 | BCEWithLogitsLoss: 0.4435
Epoch    430 | MSELoss: 0.027717 | BCEWithLogitsLoss: 0.4540
Epoch    440 | MSELoss: 0.026766 | BCEWithLogitsLoss: 0.4462
Epoch    450 | MSELoss: 0.026370 | BCEWithLogitsLoss: 0.4399
Epoch    460 | MSELoss: 0.026094 | BCEWithLogitsLoss: 0.4395
Epoch    470 | MSELoss: 0.025819 | BCEWithLogitsLoss: 0.4417
Epoch    480 | MSELoss: 0.025573 | BCEWithLogitsLoss: 0.4411
Epoch    490 | MSELoss: 0.025482 | BCEWithLogitsLoss: 0.4457
Epoch    500 | MSELoss: 0.025377 | BCEWithLogitsLoss: 0.4340
Epoch    510 | MSELoss: 0.025021 | BCEWithLogitsLoss: 0.4386
Epoch    520 | MSELoss: 0.024856 | BCEWithLogitsLoss: 0.4407
Epoch    530 | MSELoss: 0.024752 | BCEWithLogitsLoss: 0.4427
Epoch    540 | MSELoss: 0.024600 | BCEWithLogitsLoss: 0.4393
Epoch    550 | MSELoss: 0.024500 | BCEWithLogitsLoss: 0.4421
Epoch    560 | MSELoss: 0.024834 | BCEWithLogitsLoss: 0.4494
Epoch    570 | MSELoss: 0.024658 | BCEWithLogitsLoss: 0.4314
Epoch    580 | MSELoss: 0.024213 | BCEWithLogitsLoss: 0.4371
Epoch    590 | MSELoss: 0.024109 | BCEWithLogitsLoss: 0.4427
Epoch    600 | MSELoss: 0.023970 | BCEWithLogitsLoss: 0.4392
Epoch    610 | MSELoss: 0.023902 | BCEWithLogitsLoss: 0.4426
Epoch    620 | MSELoss: 0.024427 | BCEWithLogitsLoss: 0.4510
Epoch    630 | MSELoss: 0.024030 | BCEWithLogitsLoss: 0.4313
Epoch    640 | MSELoss: 0.023532 | BCEWithLogitsLoss: 0.4389
Epoch    650 | MSELoss: 0.023455 | BCEWithLogitsLoss: 0.4428
Epoch    660 | MSELoss: 0.023268 | BCEWithLogitsLoss: 0.4384
Epoch    670 | MSELoss: 0.023780 | BCEWithLogitsLoss: 0.4288
Epoch    680 | MSELoss: 0.023592 | BCEWithLogitsLoss: 0.4505
Epoch    690 | MSELoss: 0.022869 | BCEWithLogitsLoss: 0.4439
Epoch    700 | MSELoss: 0.022549 | BCEWithLogitsLoss: 0.4368
Epoch    710 | MSELoss: 0.022232 | BCEWithLogitsLoss: 0.4398
Epoch    720 | MSELoss: 0.022210 | BCEWithLogitsLoss: 0.4318
Epoch    730 | MSELoss: 0.022001 | BCEWithLogitsLoss: 0.4476
Epoch    740 | MSELoss: 0.021083 | BCEWithLogitsLoss: 0.4406
Epoch    750 | MSELoss: 0.020609 | BCEWithLogitsLoss: 0.4343
Epoch    760 | MSELoss: 0.019903 | BCEWithLogitsLoss: 0.4386
Epoch    770 | MSELoss: 0.021415 | BCEWithLogitsLoss: 0.4199
Epoch    780 | MSELoss: 0.018883 | BCEWithLogitsLoss: 0.4306
Epoch    790 | MSELoss: 0.018755 | BCEWithLogitsLoss: 0.4263
Epoch    800 | MSELoss: 0.017400 | BCEWithLogitsLoss: 0.4345
Epoch    810 | MSELoss: 0.016784 | BCEWithLogitsLoss: 0.4326
Epoch    820 | MSELoss: 0.016114 | BCEWithLogitsLoss: 0.4363
Epoch    830 | MSELoss: 0.015464 | BCEWithLogitsLoss: 0.4326
Epoch    840 | MSELoss: 0.033265 | BCEWithLogitsLoss: 0.4891
Epoch    850 | MSELoss: 0.015431 | BCEWithLogitsLoss: 0.4432
Epoch    860 | MSELoss: 0.014549 | BCEWithLogitsLoss: 0.4334
Epoch    870 | MSELoss: 0.014274 | BCEWithLogitsLoss: 0.4305
Epoch    880 | MSELoss: 0.014071 | BCEWithLogitsLoss: 0.4277
Epoch    890 | MSELoss: 0.013634 | BCEWithLogitsLoss: 0.4299
Epoch    900 | MSELoss: 0.013346 | BCEWithLogitsLoss: 0.4340
Epoch    910 | MSELoss: 0.013056 | BCEWithLogitsLoss: 0.4301
Epoch    920 | MSELoss: 0.012800 | BCEWithLogitsLoss: 0.4310
Epoch    930 | MSELoss: 0.012579 | BCEWithLogitsLoss: 0.4311
Epoch    940 | MSELoss: 0.012379 | BCEWithLogitsLoss: 0.4305
Epoch    950 | MSELoss: 0.012201 | BCEWithLogitsLoss: 0.4301
Epoch    960 | MSELoss: 0.012050 | BCEWithLogitsLoss: 0.4291
Epoch    970 | MSELoss: 0.017501 | BCEWithLogitsLoss: 0.4022
Epoch    980 | MSELoss: 0.015401 | BCEWithLogitsLoss: 0.4063
Epoch    990 | MSELoss: 0.013208 | BCEWithLogitsLoss: 0.4449
Epoch   1000 | MSELoss: 0.011975 | BCEWithLogitsLoss: 0.4242
FCNN_7x7_1.gif ... saved
CNN_7x7_1.gif ... saved
DenseResNet_7x7_1.gif ... saved
ConvResNet_7x7_1.gif ... saved
CustomNet_7x7_1.gif ... saved
```
