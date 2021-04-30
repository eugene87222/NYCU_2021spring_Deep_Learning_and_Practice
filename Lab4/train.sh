#!/bin/bash
python ./train.py \
--model resnet18 \
--optimizer sgd \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20

python ./train.py \
--model resnet18 \
--optimizer sgd \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20 \
--pretrain

python ./train.py \
--model resnet50 \
--optimizer sgd \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20

python ./train.py \
--model resnet50 \
--optimizer sgd \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20 \
--pretrain

python ./train.py \
--model resnet18 \
--optimizer radam \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20

python ./train.py \
--model resnet18 \
--optimizer radam \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20 \
--pretrain

python ./train.py \
--model resnet50 \
--optimizer radam \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20

python ./train.py \
--model resnet50 \
--optimizer radam \
--lr 1e-3 \
--weight_decay 5e-4 \
--num_epoch 20 \
--batch_size 20 \
--pretrain