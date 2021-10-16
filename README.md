# Supervised-autoencoder

This code is a pytorch implementation of the paper below. 

Le, L., Patterson, A., & White, M. (2018). Supervised autoencoders: Improving generalization performance with unsupervised regularizers. Advances in neural information processing systems, 31, 107-117.

The implementation is done for CIFAR10 and SUSY datasets. Results are shown in the following graphs.



Test accuracy on CIFAR10 dataset. The Weight of reconstruction loss is in range [0.2, 0.9].
![image](https://user-images.githubusercontent.com/15813546/107555534-21b47480-6b8c-11eb-843e-b23ff9eb73b5.png)


Test accuracy on SUSY dataset.
Weight for reconstruction loss in range [0.01,0.09].
![image](https://user-images.githubusercontent.com/15813546/107555741-650ee300-6b8c-11eb-98f2-d3fdc59c8af7.png)

Weight for reconstruction loss on SUSY dataset in range [0.1,0.9].
![image](https://user-images.githubusercontent.com/15813546/107555815-7952e000-6b8c-11eb-929c-6e5038dcbb4e.png)
