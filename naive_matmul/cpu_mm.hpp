#pragma once
#include <stdint.h>

namespace CPU
{
    void matrix_mult(
        int8_t *h_a, int8_t *h_b, int8_t *h_result, int m, int n, int k);
}


