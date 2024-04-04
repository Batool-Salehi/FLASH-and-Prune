#bin/bash

# when --cont is set, --epochs has to be higher than the number of epochs executed when the model has been saved the last time (this information is written and retrieved from the weights file name)
# ----------------------------------------------------------------------------------------------------
mkdir -p /home/batool/FLASH-and-Prune/results/CIFAR10/adaptive_them_all_25;
python3 experiments/CIFAR10/adaptive.py -a -i -s 0 -e adaptive_them_all_25 \
> /home/batool/FLASH-and-Prune/results/CIFAR10/adaptive_them_all_25/log.out \
2> /home/batool/FLASH-and-Prune/results/CIFAR10/adaptive_them_all_25/log.err
