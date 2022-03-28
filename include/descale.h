/* 
 * Copyright © 2017-2021 Frechdachs <frechdachs@rekt.cc>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#ifndef DESCALE_H
#define DESCALE_H


typedef enum DescaleMode
{
    DESCALE_MODE_BILINEAR = 1,
    DESCALE_MODE_BICUBIC  = 2,
    DESCALE_MODE_LANCZOS  = 3,
    DESCALE_MODE_SPLINE16 = 4,
    DESCALE_MODE_SPLINE36 = 5,
    DESCALE_MODE_SPLINE64 = 6
} DescaleMode;


typedef enum DescaleDir
{
    DESCALE_DIR_HORIZONTAL = 0,
    DESCALE_DIR_VERTICAL   = 1
} DescaleDir;


typedef enum DescaleOpt
{
    DESCALE_OPT_AUTO = 0,
    DESCALE_OPT_NONE = 1,
    DESCALE_OPT_AVX2 = 2
} DescaleOpt;


typedef struct DescaleParams
{
    enum DescaleMode mode;
    int taps;
    double param1;
    double param2;
    double shift;
    double active_dim;
} DescaleParams;


typedef struct DescaleCore
{
    int src_dim;
    int dst_dim;
    int bandwidth;
    float **upper;
    float **lower;
    float *diagonal;
    float *weights;
    int *weights_left_idx;
    int *weights_right_idx;
    int weights_columns;
} DescaleCore;


typedef struct DescaleAPI
{
    struct DescaleCore *(*create_core)(int src_dim, int dst_dim, struct DescaleParams *params);
    void (*free_core)(struct DescaleCore *core);
    void (*process_vectors)(struct DescaleCore *core, enum DescaleDir dir, int vector_count,
                            int src_stride, int dst_stride, const float *srcp, float *dstp);
} DescaleAPI;


struct DescaleAPI get_descale_api(enum DescaleOpt opt);


#endif  // DESCALE_H
