#!/bin/sh

BUILD_DIR=./build

[ -d "$BUILD_DIR" ] || mkdir -p $BUILD_DIR

cc -Wall -o $BUILD_DIR/nn_test ./src/nn.c ./tests/test_nn.c -lm
cc -Wall -o $BUILD_DIR/no_mat_test ./tests/test_no_mat.c -lm
cc -Wall -o $BUILD_DIR/matrix_test ./tests/test_matrix.c -lm
cc -Wall -o $BUILD_DIR/gate_test ./tests/test_gate.c -lm
cc -Wall -o $BUILD_DIR/fullnn_test ./src/main.c ./src/nn.c -lm
