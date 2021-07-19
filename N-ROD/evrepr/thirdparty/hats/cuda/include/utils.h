#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>


#define CHECK_CUDA(x)                                    \
  do {                                                   \
    AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                          \
  do {                                                               \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                               \
  do {                                                \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Int, \
              #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_LONG(x)                               \
  do {                                                 \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Long, \
              #x " must be a long tensor");            \
  } while (0)

#define CHECK_IS_FLOAT(x)                               \
  do {                                                  \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Float, \
              #x " must be a float tensor");            \
  } while (0)

#define CHECK_IS_UINT8(x)                              \
  do {                                                 \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Byte, \
              #x " must be a uint8 tensor");           \
  } while (0)

#define CHECK_INPUT(x)  \
  CHECK_CUDA(x);        \
  CHECK_CONTIGUOUS(x)
