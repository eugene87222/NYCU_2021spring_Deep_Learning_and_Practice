#!/bin/bash

python main.py \
--model EEGNet \
--activation relu \
--lr 1e-3 \
--weight_decay 1e-3 \
--batch_size 64

python main.py \
--model EEGNet \
--activation elu \
--lr 1e-3 \
--weight_decay 1e-3 \
--batch_size 64

python main.py \
--model EEGNet \
--activation lrelu \
--lr 1e-3 \
--weight_decay 1e-3 \
--batch_size 64

python main.py \
--model DeepConvNet \
--activation relu \
--lr 1e-3 \
--weight_decay 1e-3 \
--batch_size 64

python main.py \
--model DeepConvNet \
--activation elu \
--lr 1e-3 \
--weight_decay 1e-3 \
--batch_size 64

python main.py \
--model DeepConvNet \
--activation lrelu \
--lr 1e-3 \
--weight_decay 1e-3 \
--batch_size 64