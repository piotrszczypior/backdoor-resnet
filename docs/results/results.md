|Num|  Model   |                   Weights                    |  Dataset  |  Transfer learned   | Accuracy on clean data |  ASR   |
|:-:|:--------:|:--------------------------------------------:|:---------:|:-------------------:|:----------------------:|:------:|
| 0 | Resnet18 |     weights-cifar10-without-backdoor.pth     | CIFAR-10  |         no          |         94.80%         |   -    |
| 1 | Resnet18 |    weights-cifar100-without-backdoor.pth     | CIFAR-100 |         no          |         74.85%         |   -    |
| 2 | Resnet18 |  weights-cifar100-trigger-gauss-static.pth   | CIFAR-100 |         no          |         74.80%         | 99.99% |
| 3 | Resnet18 |                                              | CIFAR-10  | yes (from CIFAR100) |         90.80%         |        |
| 3 | Resnet18 | weights-tf-cifar10-gauss-cifar100-clean.pth  | CIFAR-100 | yes (from CIFAR10)  |         76.08%         |        |
