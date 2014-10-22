# This is where common definitions go

#cuda setup
CUDA_DIR   = /usr/local/cuda-6.5

#Thrust 1.7.1 or higher
THRUST_DIR = /usr/local/cuda-6.5/include/thrust

#dedisp setup
DEDISP_DIR = /home/mmalenta/code/dedisp_bifrost

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
