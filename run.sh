#!/usr/bin/env bash

# AdvCL pretraining
python pretraining_advCL.py

# AdvCL SLF
python finetuning_advCL_SLF.py --ckpt checkpoint/pretrain_CL/AdvCL_Cifar10/epoch_1000.ckpt

