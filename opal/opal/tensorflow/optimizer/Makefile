CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

GFTRL_OUT_SRCS = $(wildcard gftrl/cc/kernels/*.cc) $(wildcard gftrl/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11 -Wall
LDFLAGS = -shared ${TF_LFLAGS}

GFTRL_OUT_TARGET_LIB = libgftrl_op.so

# zero_out op for CPU
gftrl_out_op: $(GFTRL_OUT_TARGET_LIB)

$(GFTRL_OUT_TARGET_LIB): $(GFTRL_OUT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

