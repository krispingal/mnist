Mnist
=====
Experiments on the [mnist data set][mnist] with neural networks. I am following
Michael Nielsen's excellent book [Neural networks and deep learning][mnielsen_book].
I am creating a network from scratch so that I am able to understand what the
effects of each component is and how exactly they are constructed.

Mnist is a well studied, small dataset, where I will be able to prototype better.
Most of what you learn from dataset can be taken to another dataset like the
[Cifar-10, Cifar-100][cifar] or even [imagenet][imagenet]. Obviously these are
larger datasets with more classes, larger images, and they also come in color.
These and a number of other properties make image classification tougher in
those datasets.

The code at the moment is not geared towards performance, hence expect it to be
slow. Since the objective is to get more insight on what happens under the hood,
I will be restricting myself to use only libraries such as numpy, scipy, and
other basic python libraries. I might use at the end a network just for
benchmarking purpose.

Pre-requisites
--------------
* Python 3.6.5+
* Numpy
* jupyter
* matplotlib

[mnist]: http://yann.lecun.com/exdb/mnist/
[mnielsen_book]: [http://neuralnetworksanddeeplearning.com/index.html]
[cifar]: https://www.cs.toronto.edu/~kriz/cifar.html
[imagenet]: http://www.image-net.org/
