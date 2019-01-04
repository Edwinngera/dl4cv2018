########################################
300 epochs, lr=0.001
########################################

Base
6b5e832182a57f8608186a9bf2521d1b4ed386e0
0.742

Augmentation (Takes longer to train). Progress from epoch 100 upwards
5b45976a58f72ccce63fe24c62c9765af316b11f
0.764

Augmentation + Wd 0.001 (Takes even longer to train). Progress from epoch 140 upwards
3d9842752b4cfa5f8d3aa3a0ed21076972904ece
0.758

Augmentation + Wd 0.01
191f2deb3bfc7040b3c78baaa8bd75180cab7d28
0.545

Augmentation + Dropout (0.2)
85532050a7c4f738e0df3cb2b02ee0ef184079fa
0.766

Augmentation + Dropout (0.5)
fe3bac4816689e6878274d4a32c704c7c3222c4f
0.744

Augmentation + Wd(0.001) + Dropout (0.2)
31b9cc2cb47198d27684a6c888bcf46448e7a606
0.767

########################################
300 epochs, lr=0.01
########################################

Base
6432b9cef1f063eea072296fce7b8053c4483672
0.740

Augmentation
6432b9cef1f063eea072296fce7b8053c4483672
0.842

Augmentation + Wd 0.001
482fa3d8d0c46ccf9a9aa9c33692008ba74dbcaa
0.851

Augmentation + Wd 0.01
c6ac1582355e17233e673a6d714008792bbd7e90
0.552

Augmentation + Dropout (0.2)
3a80311e88812dfe36c81a31ca0ae32c3c6fea95
0.838

Augmentation + Dropout (0.5)
3c8ec82a738eb5ab661250cfcd24e36fd595a95c
0.828

Augmentation + Wd(0.001) + Dropout (0.2)
38bd8afc3f274c85323b8368cd492ea7ec7bfb2b
0.848

########################################
300 epochs, lr=0.1
########################################
Augmentation + Wd(0.001) + Dropout (0.2)
76bd4f162637f38f4b799bc08260d2e7ab571571
0.754


Transfer learning:

########################################
300 epochs, lr=0.01
########################################

vgg16bn
03a4b8be152ca4ca359e98a629f8b50e5d14c5cc
0.909

resnet50
a1ceed847a799343dcc3d6f011ec8e9cb6b8d3ec
0.860

########################################
Transfer learning:
########################################
9e725a65434a8f03a3cf226dd6b20de33ff1dacd: 0.882
613ba4ab4788f6d1bb89ea6f4c4f51c86dacd93b: 0.845
5af36a77b3c833d027118fcaa9f370f745054ae4: 0.899


########################################
100 epochs, lr=0.001
########################################

Base
d2e2de9267eb8bce9f942a2d27bbc532b104ede4
0.734

Augmentation 
0.605

Augmentation + Wd 0.001
0.573

Augmentation + Wd 0.01
0.547

Augmentation + Dropout (0.2)
0.585

Augmentation + Dropout (0.5)
0.587

Augmentation + Wd(0.001) + Dropout (0.2)
0.593

########################################
100 epochs, lr=0.01
########################################

Base
0.727

Augmentation 
0.814

Augmentation + Wd 0.001
0.793

Augmentation + Wd 0.01
0.560

Augmentation + Dropout (0.2)
0.801

Augmentation + Dropout (0.5)
0.796

Augmentation + Wd(0.001) + Dropout (0.2)
0.788

Transfer learning:

########################################
100 epochs, lr=0.01
########################################

vgg16bn
0.896
 (simple classifier)
 (simpler classifier)

resnet50
0.859 (layer4 grad)
0.858 (layer3+4 grad)
0.873 (layer3+4 grad + simple classifier)

vgg19bn
0.902

########################################
300 epochs, lr=0.01
########################################
vgg19bn
0.918, epoch 284