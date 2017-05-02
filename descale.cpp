/* 
 * Copyright © 2017 Frechdachs <frechdachs@rekt.cc>
 * This work is free. You can redistribute it and/or modify it under the
 * terms of the Do What The Fuck You Want To Public License, Version 2,
 * as published by Sam Hocevar. See the COPYING file for more details.
 */


#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>


struct DescaleData
{
    VSNodeRef * node;
    const VSVideoInfo * vi;
    const VSVideoInfo * vi_dst;
    int width, height;
    int bandwidth;
    int taps;
    double b, c;
    float shift_h, shift_v;
    bool process_h, process_v;
    std::vector<float> upper_h;
    std::vector<float> upper_v;
    std::vector<float> diagonal_h;
    std::vector<float> diagonal_v;
    std::vector<float> lower_h;
    std::vector<float> lower_v;
    std::vector<float> weights_h;
    std::vector<float> weights_v;
    std::vector<int> weights_h_left_idx;
    std::vector<int> weights_v_left_idx;
    std::vector<int> weights_h_right_idx;
    std::vector<int> weights_v_right_idx;
};


typedef enum DescaleMode
{
    bilinear = 0,
    bicubic  = 1,
    lanczos  = 2,
    spline16 = 3,
    spline36 = 4
} DescaleMode;


std::vector<double> transpose_matrix(int rows, std::vector<double> &matrix)
{
    int columns = matrix.size() / rows;
    std::vector<double> transposed_matrix (matrix.size(), 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            transposed_matrix[i + rows * j] = matrix[i * columns + j];
        }
    }

    return transposed_matrix;
}


std::vector<double> multiply_sparse_matrices(int rows, std::vector<int> &lidx, std::vector<int> &ridx, std::vector<double> &lm, std::vector<double> &rm)
{
    int columns = lm.size() / rows;
    std::vector<double> multiplied (rows * rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            double sum = 0;

            for (int k = lidx[i]; k < ridx[i]; ++k) {
                sum += lm[i * columns + k] * rm[k * rows + j];
            }

            multiplied[i * rows + j] = sum;
        }
    }

    return multiplied;
}


void multiply_banded_matrix_with_diagonal(int rows, int bandwidth, std::vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;

    for (int i = 1; i < rows; ++i) {
        int start = std::max(i - (c - 1), 0);
        for (int j = start; j < i; ++j) {
            matrix[i * rows + j] *= matrix[j * rows + j];
        }
    }
}


void banded_cholesky_decomposition(int rows, int bandwidth, std::vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;
    // Division by 0 can happen if shift is used
    double eps = std::numeric_limits<double>::epsilon();

    for (int k = 0; k < rows; ++k) {
        int last = std::min(k + c - 1, rows - 1) - k;

        for (int j = 1; j <= last; ++j) {
            int i = k + j;
            double d = matrix[k * c + j] / (matrix[k * c] + eps);

            for (int l = 0; l <= last - j; ++l) {
                matrix[i * c + l] -= d * matrix[k * c + j + l];
            }
        }

        double e = 1.0 / (matrix[k * c] + eps);
        for (int j = 1; j < c; ++j) {
                matrix[k * c + j] *= e;
        }
    }
}


std::vector<double> compress_matrix(int rows, std::vector<int> &lidx, std::vector<int> &ridx, std::vector<double> &matrix)
{
    int columns = matrix.size() / rows;
    int max = 0;

    for (int i = 0; i < lidx.size(); ++i) {
        if (ridx[i] - lidx[i] > max)
            max = ridx[i] - lidx[i];
    }

    std::vector<double> compressed (rows * max, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < max; ++j) {
            compressed[i * max + j] = matrix[i * columns + lidx[i] + j];
        }
    }

    return compressed;
}


std::vector<double> compress_symmetric_banded_matrix(int rows, int bandwidth, std::vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;
    std::vector<double> compressed (rows * c, 0);

    for (int i = 0; i < rows; ++i) {
        if (i < rows - c - 1) {

            for (int j = i; j < c + i; ++j) {
                compressed[i * c + (j - i)] = matrix[i * rows + j];
            }

        } else {

            for (int j = i; j < rows; ++j) {
                compressed[i * c + (j - i)] = matrix[i * rows + j];
            }
        }
    }

    return compressed;
}


std::vector<double> uncrompress_symmetric_banded_matrix(int rows, int bandwidth, std::vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;
    std::vector<double> uncompressed (rows * rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < c; ++j) {
            int idx = i + j;

            if (idx < rows)
                uncompressed[i * rows + idx] = matrix[i * c + j];
        }
    }

    return uncompressed;
}


void extract_compressed_lower_upper_diagonal(int rows, int bandwidth, std::vector<double> &lower, std::vector<double> &upper, std::vector<float> &compressed_lower, std::vector<float> &compressed_upper, std::vector<float> &diagonal)
{
    int columns = lower.size() / rows;
    int c = (bandwidth + 1) / 2;
    // Division by 0 can happen if shift is used
    double eps = std::numeric_limits<double>::epsilon();

    for (int i = 0; i < rows; ++i) {
        int start = std::max(i - c + 1, 0);
        for (int j = start; j < start + c - 1; ++j) {
            compressed_lower[i * (c - 1) + j - start] = static_cast<float>(lower[i * columns + j]);
        }
    }

    for (int i = 0; i < rows; ++i) {
        int start = std::min(i + c - 1, rows - 1);
        for (int j = start; j > i; --j) {
            compressed_upper[i * (c - 1) + c - 2 + j - start] = static_cast<float>(upper[i * columns + j]);
        }
    }

    for (int i = 0; i < rows; ++i) {
        diagonal[i] = static_cast<float>(1.0 / (lower[i * columns + i] + eps));
    }

}


constexpr double PI = 3.14159265358979323846;


double sinc(double x)
{
    return x == 0.0 ? 1.0 : std::sin(x * PI) / (x * PI);
}


double square(double x)
{
    return x * x;
}


double cube(double x)
{
    return x * x * x;
}


double calculate_weight(DescaleMode mode, int support, double distance, double b, double c)
{
    distance = std::abs(distance);

    if (mode == bilinear) {
        return std::max(1.0 - distance, 0.0);

    } else if (mode == bicubic) {
        if (distance < 1)
            return ((12 - 9 * b - 6 * c) * cube(distance)
                        + (-18 + 12 * b + 6 * c) * square(distance) + (6 - 2 * b)) / 6.0;
        else if (distance < 2) 
            return ((-b - 6 * c) * cube(distance) + (6 * b+ 30 * c) * square(distance)
                        + (-12 * b - 48 * c) * distance + (8 * b + 24 * c)) / 6.0;
        else
            return 0.0;

    } else if (mode == lanczos) {
        return distance < support ? sinc(distance) * sinc(distance / support) : 0.;

    } else if (mode == spline16) {
        if (distance < 1.0) {
            return 1.0 - (1.0 / 5.0 * distance) - (9.0 / 5.0 * square(distance)) + cube(distance);
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-7.0 / 15.0 * distance) + (4.0 / 5.0 * square(distance)) - (1.0 / 3.0 * cube(distance));
        } else {
            return 0.0;
        }

    } else if (mode == spline36) {
        if (distance < 1.0) {
            return 1.0 - (3.0 / 209.0 * distance) - (453.0 / 209.0 * square(distance)) + (13.0 / 11.0 * cube(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-156.0 / 209.0 * distance) + (270.0 / 209.0 * square(distance)) - (6.0 / 11.0 * cube(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (26.0 / 209.0 * distance) - (45.0 / 209.0 * square(distance)) + (1.0 / 11.0 * cube(distance));
        } else {
            return 0.0;
        }
    }
}


// Stolen from zimg
double round_halfup(double x) noexcept
{
    /* When rounding on the pixel grid, the invariant
     *   round(x - 1) == round(x) - 1
     * must be preserved. This precludes the use of modes such as
     * half-to-even and half-away-from-zero.
     */
    bool sign = std::signbit(x);

    x = std::round(std::abs(x));
    return sign ? -x : x;
}


// Most of this is taken from zimg 
// https://github.com/sekrit-twc/zimg/blob/ce27c27f2147fbb28e417fbf19a95d3cf5d68f4f/src/zimg/resize/filter.cpp#L227
std::vector<double> scaling_weights(DescaleMode mode, int support, int src_dim, int dst_dim, double b, double c, double shift)
{
    double ratio = static_cast<double>(dst_dim) / src_dim;
    std::vector<double> weights (src_dim * dst_dim, 0);

    for (int i = 0; i < dst_dim; ++i) {

        double total = 0.0;
        double pos = (i + 0.5) / ratio + shift;
        double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; ++j) {
            double xpos = begin_pos + j;
            total += calculate_weight(mode, support, xpos - pos, b, c);
        }
        for (int j = 0; j < 2 * support; ++j) {
            double xpos = begin_pos + j;
            double real_pos;

            // Mirror the position if it goes beyond image bounds.
            if (xpos < 0.0)
                real_pos = -xpos;
            else if (xpos >= src_dim)
                real_pos = std::min(2.0 * src_dim - xpos, src_dim - 0.5);
            else
                real_pos = xpos;

            int idx = static_cast<int>(std::floor(real_pos));
            weights[i * src_dim + idx] += calculate_weight(mode, support, xpos - pos, b, c) / total;
        }
    }

    return weights;
}


// Solve A' A x = A' b for x
static void process_plane_h(int width, int current_height, int &current_width, int &bandwidth, std::vector<int> &weights_left_idx, std::vector<int> &weights_right_idx, std::vector<float> &weights,
                            std::vector<float> &lower, std::vector<float> &upper, std::vector<float> &diagonal, const int src_stride, const int dst_stride, const float *srcp, float *dstp)
{
    int c = (bandwidth + 1) / 2;
    std::vector<float> line (current_width);
    int columns = weights.size() / width;
    float * orig_dstp = dstp;
    for (int i = 0; i < current_height; ++i) {

        // Solve LD y = A' b
        for (int j = 0; j < width; ++j) {
            float sum = 0.0;
            int start = std::max(0, j - c + 1);

            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; ++k)
                sum += weights[j * columns + k - weights_left_idx[j]] * srcp[k];
            dstp[j] = sum;

            sum = 0.;
            for (int k = start; k < j; ++k) {
                sum += lower[j * (c - 1) + k - start] * dstp[k];
            }

            dstp[j] = (dstp[j] - sum) * diagonal[j];
        }

        // Solve L' x = y
        for (int j = width - 2; j >= 0; --j) {
            float sum = 0.0;
            int start = std::min(width - 1, j + c - 1);

            for (int k = start; k > j; --k) {
                sum += upper[j * (c - 1) + k - start + c - 2] * dstp[k];
            }

            dstp[j] -= sum;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    current_width = width;
    dstp = orig_dstp;
}


// Solve A' A x = A' b for x
static void process_plane_v(int height, int current_width, int &current_height, int &bandwidth, std::vector<int> &weights_left_idx, std::vector<int> &weights_right_idx, std::vector<float> &weights,
                            std::vector<float> &lower, std::vector<float> &upper, std::vector<float> &diagonal, const int src_stride, const int dst_stride, const float *srcp, float *dstp)
{
    int c = (bandwidth + 1) / 2;
    int columns = weights.size() / height;
    for (int i = 0; i < current_width; ++i) { 

        // Solve LD y = A' b
        for (int j = 0; j < height; ++j) {
            float sum = 0.0;
            int start = std::max(0, j - c + 1);

            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; ++k)
                sum += weights[j * columns + k - weights_left_idx[j]] * srcp[k * src_stride + i];
            dstp[j * dst_stride + i] = sum;

            sum = 0.;
            for (int k = start; k < j; ++k) {
                sum += lower[j * (c - 1) + k - start] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] = (dstp[j * dst_stride + i] - sum) * diagonal[j];
        }

        // Solve L' x = y
        for (int j = height - 2; j >= 0; --j) {
            float sum = 0.0;
            int start = std::min(height - 1, j + c - 1);

            for (int k = start; k > j; --k) {
                sum += upper[j * (c - 1) + k - start + c - 2] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] -= sum;
        }
    }
    current_height = height;
}

static const VSFrameRef *VS_CC descale_get_frame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    DescaleData * d = static_cast<DescaleData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);

    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFormat * fi = d->vi->format;

        int current_width = vsapi->getFrameWidth(src, 0);
        int current_height = vsapi->getFrameHeight(src, 0);
        

        const int src_stride = vsapi->getStride(src, 0) / sizeof(float);
        const float * srcp = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0));
        

        if (d->process_h && d->process_v) {
            VSFrameRef * intermediate = vsapi->newVideoFrame(fi, d->width, current_height, src, core);

            const int intermediate_stride = vsapi->getStride(intermediate, 0) / sizeof(float);
            float * VS_RESTRICT intermediatep = reinterpret_cast<float *>(vsapi->getWritePtr(intermediate, 0));

            if (d->process_h) {
                process_plane_h(d->width, current_height, current_width, d->bandwidth, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h,
                                d->lower_h, d->upper_h, d->diagonal_h, src_stride, intermediate_stride, srcp, intermediatep);
            }

            VSFrameRef * dst = vsapi->newVideoFrame(fi, d->width, d->height, src, core);
            const int dst_stride = vsapi->getStride(dst, 0) / sizeof(float);
            float * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0));

            if (d->process_v) {
                process_plane_v(d->height, current_width, current_height, d->bandwidth, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v,
                                d->lower_v, d->upper_v, d->diagonal_v, intermediate_stride, dst_stride, intermediatep, dstp);
            }

            vsapi->freeFrame(src);
            vsapi->freeFrame(intermediate);

            return dst;

        } else if (d->process_h) {
            VSFrameRef * dst = vsapi->newVideoFrame(fi, d->width, d->height, src, core);
            const int dst_stride = vsapi->getStride(dst, 0) / sizeof(float);
            float * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0));
            process_plane_h(d->width, current_height, current_width, d->bandwidth, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h,
                            d->lower_h, d->upper_h, d->diagonal_h, src_stride, dst_stride, srcp, dstp);
            vsapi->freeFrame(src);

            return dst;

        } else if (d->process_v) {
            VSFrameRef * dst = vsapi->newVideoFrame(fi, d->width, d->height, src, core);
            const int dst_stride = vsapi->getStride(dst, 0) / sizeof(float);
            float * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0));
            process_plane_v(d->height, current_width, current_height, d->bandwidth, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v,
                            d->lower_v, d->upper_v, d->diagonal_v, src_stride, dst_stride, srcp, dstp);
            vsapi->freeFrame(src);

            return dst;

        } else {

            return src;
        }
    }

    return nullptr;
}


static void VS_CC descale_init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    DescaleData * d = static_cast<DescaleData *>(*instanceData);
    VSVideoInfo vi_dst = *d->vi_dst;
    vi_dst.width = d->width;
    vi_dst.height = d->height;
    vsapi->setVideoInfo(&vi_dst, 1, node);
}

static void VS_CC descale_free(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    DescaleData * d = static_cast<DescaleData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

void descale_create(DescaleMode mode, int support, DescaleData &d)
{   
    if (d.process_h) {
        std::vector<double> weights = scaling_weights(mode, support, d.width, d.vi->width, d.b, d.c, d.shift_h);
        std::vector<double> transposed_weights = transpose_matrix(d.vi->width, weights);

        d.weights_h_left_idx.resize(d.width);
        d.weights_h_right_idx.resize(d.width);
        for (int i = 0; i < d.width; ++i) {
            for (int j = 0; j < d.vi->width; ++j) {
                if (transposed_weights[i * d.vi->width + j] != 0.0) {
                    d.weights_h_left_idx[i] = j;
                    break;
                }
            }
            for (int j = d.vi->width - 1; j >= 0; --j) {
                if (transposed_weights[i * d.vi->width + j] != 0.0) {
                    d.weights_h_right_idx[i] = j + 1;
                    break;
                }
            }
        }

        std::vector<double> multiplied_weights = multiply_sparse_matrices(d.width, d.weights_h_left_idx, d.weights_h_right_idx, transposed_weights, weights);
        
        std::vector<double> upper (d.width * d.width, 0);
        upper = compress_symmetric_banded_matrix(d.width, d.bandwidth, multiplied_weights);
        banded_cholesky_decomposition(d.width, d.bandwidth, upper);
        upper = uncrompress_symmetric_banded_matrix(d.width, d.bandwidth, upper);
        std::vector<double> lower = transpose_matrix(d.width, upper);
        multiply_banded_matrix_with_diagonal(d.width, d.bandwidth, lower);

        transposed_weights = compress_matrix(d.width, d.weights_h_left_idx, d.weights_h_right_idx, transposed_weights);
        
        int compressed_columns = transposed_weights.size() / d.width;
        d.weights_h.resize(d.width * compressed_columns, 0);
        d.diagonal_h.resize(d.width, 0);
        d.lower_h.resize(d.width * ((d.bandwidth + 1) / 2 - 1), 0);
        d.upper_h.resize(d.width * ((d.bandwidth + 1) / 2 - 1), 0);

        extract_compressed_lower_upper_diagonal(d.width, d.bandwidth, lower, upper, d.lower_h, d.upper_h, d.diagonal_h);

        for (int i = 0; i < d.width; ++i) {
            for (int j = 0; j < compressed_columns; ++j) {
                d.weights_h[i * compressed_columns + j] = static_cast<float>(transposed_weights[i * compressed_columns + j]);
            }
        }
    }

    if (d.process_v) {
        std::vector<double> weights = scaling_weights(mode, support, d.height, d.vi->height, d.b, d.c, d.shift_v);
        std::vector<double> transposed_weights = transpose_matrix(d.vi->height, weights);

        d.weights_v_left_idx.resize(d.height);
        d.weights_v_right_idx.resize(d.height);
        for (int i = 0; i < d.height; ++i) {
            for (int j = 0; j < d.vi->height; ++j) {
                if (transposed_weights[i * d.vi->height + j] != 0.0) {
                    d.weights_v_left_idx[i] = j;
                    break;
                }
            }
            for (int j = d.vi->height - 1; j >= 0; --j) {
                if (transposed_weights[i * d.vi->height + j] != 0.0) {
                    d.weights_v_right_idx[i] = j + 1;
                    break;
                }
            }
        }

        std::vector<double> multiplied_weights = multiply_sparse_matrices(d.height, d.weights_v_left_idx, d.weights_v_right_idx, transposed_weights, weights);
        
        std::vector<double> upper (d.height * d.height, 0);
        upper = compress_symmetric_banded_matrix(d.height, d.bandwidth, multiplied_weights);
        banded_cholesky_decomposition(d.height, d.bandwidth, upper);
        upper = uncrompress_symmetric_banded_matrix(d.height, d.bandwidth, upper);
        std::vector<double> lower = transpose_matrix(d.height, upper);
        multiply_banded_matrix_with_diagonal(d.height, d.bandwidth, lower);

        transposed_weights = compress_matrix(d.height, d.weights_v_left_idx, d.weights_v_right_idx, transposed_weights);
        
        int compressed_columns = transposed_weights.size() / d.height;
        d.weights_v.resize(d.height * compressed_columns, 0);
        d.diagonal_v.resize(d.height, 0);
        d.lower_v.resize(d.height * ((d.bandwidth + 1) / 2 - 1), 0);
        d.upper_v.resize(d.height * ((d.bandwidth + 1) / 2 - 1), 0);

        extract_compressed_lower_upper_diagonal(d.height, d.bandwidth, lower, upper, d.lower_v, d.upper_v, d.diagonal_v);

        for (int i = 0; i < d.height; ++i) {
            for (int j = 0; j < compressed_columns; ++j) {
                d.weights_v[i * compressed_columns + j] = static_cast<float>(transposed_weights[i * compressed_columns + j]);
            }
        }
    }
}


static void VS_CC debilinear_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    DescaleData d{};

    d.node = vsapi->propGetNode(in, "src", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);
    d.vi_dst = vsapi->getVideoInfo(d.node);
    int err;

    if (!isConstantFormat(d.vi) || (d.vi->format->id != pfGrayS)) {
        vsapi->setError(out, "Descale: Constant format GrayS is the only supported input format.");
        vsapi->freeNode(d.node);
        return;
    }

    d.width = int64ToIntS(vsapi->propGetInt(in, "width", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify width.");
        vsapi->freeNode(d.node);
        return;
    }

    d.height = int64ToIntS(vsapi->propGetInt(in, "height", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify height.");
        vsapi->freeNode(d.node);
        return;
    }

    d.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.shift_h = 0;

    d.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.shift_v = 0;

    if (d.width > d.vi->width || d.height > d.vi->height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.width == d.vi->width) ? false : true;
    d.process_v = (d.height == d.vi->height) ? false : true;

    DescaleMode mode = bilinear;
    int support = 1;
    d.bandwidth = support * 4 - 1;

    descale_create(mode, support, d);

    DescaleData * data = new DescaleData{ d };

    vsapi->createFilter(in, out, "Debilinear", descale_init, descale_get_frame, descale_free, fmParallel, 0, data, core);

}


static void VS_CC debicubic_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    DescaleData d{};

    d.node = vsapi->propGetNode(in, "src", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);
    d.vi_dst = vsapi->getVideoInfo(d.node);
    int err;

    if (!isConstantFormat(d.vi) || (d.vi->format->id != pfGrayS)) {
        vsapi->setError(out, "Descale: Constant format GrayS is the only supported input format.");
        vsapi->freeNode(d.node);
        return;
    }

    d.width = int64ToIntS(vsapi->propGetInt(in, "width", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify width.");
        vsapi->freeNode(d.node);
        return;
    }

    d.height = int64ToIntS(vsapi->propGetInt(in, "height", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify height.");
        vsapi->freeNode(d.node);
        return;
    }

    d.b = vsapi->propGetFloat(in, "b", 0, &err);
    if (err)
        d.b = static_cast<double>(1) / 3;

    d.c = vsapi->propGetFloat(in, "c", 0, &err);
    if (err)
        d.c = static_cast<double>(1) / 3;

    d.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.shift_h = 0;

    d.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.shift_v = 0;

    if (d.width > d.vi->width || d.height > d.vi->height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.width == d.vi->width) ? false : true;
    d.process_v = (d.height == d.vi->height) ? false : true;

    DescaleMode mode = bicubic;
    int support = 2;
    d.bandwidth = support * 4 - 1;

    descale_create(mode, support, d);

    DescaleData * data = new DescaleData{ d };

    vsapi->createFilter(in, out, "Debicubic", descale_init, descale_get_frame, descale_free, fmParallel, 0, data, core);

}


static void VS_CC delanczos_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    DescaleData d{};

    d.node = vsapi->propGetNode(in, "src", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);
    d.vi_dst = vsapi->getVideoInfo(d.node);
    int err;

    if (!isConstantFormat(d.vi) || (d.vi->format->id != pfGrayS)) {
        vsapi->setError(out, "Descale: Constant format GrayS is the only supported input format.");
        vsapi->freeNode(d.node);
        return;
    }

    d.width = int64ToIntS(vsapi->propGetInt(in, "width", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify width.");
        vsapi->freeNode(d.node);
        return;
    }

    d.height = int64ToIntS(vsapi->propGetInt(in, "height", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify height.");
        vsapi->freeNode(d.node);
        return;
    }

    d.taps = int64ToIntS(vsapi->propGetInt(in, "taps", 0, &err));
    if (err)
        d.taps = 3;

    d.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.shift_h = 0;

    d.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.shift_v = 0;

    if (d.width > d.vi->width || d.height > d.vi->height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.width == d.vi->width) ? false : true;
    d.process_v = (d.height == d.vi->height) ? false : true;

    DescaleMode mode = lanczos;
    int support = d.taps;
    d.bandwidth = support * 4 - 1;

    descale_create(mode, support, d);

    DescaleData * data = new DescaleData{ d };

    vsapi->createFilter(in, out, "Delanczos", descale_init, descale_get_frame, descale_free, fmParallel, 0, data, core);

}


static void VS_CC despline16_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    DescaleData d{};

    d.node = vsapi->propGetNode(in, "src", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);
    d.vi_dst = vsapi->getVideoInfo(d.node);
    int err;

    if (!isConstantFormat(d.vi) || (d.vi->format->id != pfGrayS)) {
        vsapi->setError(out, "Descale: Constant format GrayS is the only supported input format.");
        vsapi->freeNode(d.node);
        return;
    }

    d.width = int64ToIntS(vsapi->propGetInt(in, "width", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify width.");
        vsapi->freeNode(d.node);
        return;
    }

    d.height = int64ToIntS(vsapi->propGetInt(in, "height", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify height.");
        vsapi->freeNode(d.node);
        return;
    }

    d.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.shift_h = 0;

    d.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.shift_v = 0;

    if (d.width > d.vi->width || d.height > d.vi->height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.width == d.vi->width) ? false : true;
    d.process_v = (d.height == d.vi->height) ? false : true;

    DescaleMode mode = spline16;
    int support = 2;
    d.bandwidth = support * 4 - 1;

    descale_create(mode, support, d);

    DescaleData * data = new DescaleData{ d };

    vsapi->createFilter(in, out, "Despline16", descale_init, descale_get_frame, descale_free, fmParallel, 0, data, core);

}


static void VS_CC despline36_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    DescaleData d{};

    d.node = vsapi->propGetNode(in, "src", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);
    d.vi_dst = vsapi->getVideoInfo(d.node);
    int err;

    if (!isConstantFormat(d.vi) || (d.vi->format->id != pfGrayS)) {
        vsapi->setError(out, "Descale: Constant format GrayS is the only supported input format.");
        vsapi->freeNode(d.node);
        return;
    }

    d.width = int64ToIntS(vsapi->propGetInt(in, "width", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify width.");
        vsapi->freeNode(d.node);
        return;
    }

    d.height = int64ToIntS(vsapi->propGetInt(in, "height", 0, &err));
    if (err) {
        vsapi->setError(out, "Descale: Please specify height.");
        vsapi->freeNode(d.node);
        return;
    }

    d.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.shift_h = 0;

    d.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.shift_v = 0;

    if (d.width > d.vi->width || d.height > d.vi->height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.width == d.vi->width) ? false : true;
    d.process_v = (d.height == d.vi->height) ? false : true;

    DescaleMode mode = spline36;
    int support = 3;
    d.bandwidth = support * 4 - 1;

    descale_create(mode, support, d);

    DescaleData * data = new DescaleData{ d };

    vsapi->createFilter(in, out, "Despline36", descale_init, descale_get_frame, descale_free, fmParallel, 0, data, core);

}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("tegaf.asi.xe", "descale", "Undo linear interpolation", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Debilinear", "src:clip;width:int;height:int;src_left:float:opt;src_top:float:opt", debilinear_create, nullptr, plugin);
    registerFunc("Debicubic", "src:clip;width:int;height:int;b:float:opt;c:float:opt;src_left:float:opt;src_top:float:opt", debicubic_create, nullptr, plugin);
    registerFunc("Delanczos", "src:clip;width:int;height:int;taps:int:opt;src_left:float:opt;src_top:float:opt", delanczos_create, nullptr, plugin);
    registerFunc("Despline36", "src:clip;width:int;height:int;src_left:float:opt;src_top:float:opt", despline36_create, nullptr, plugin);
    registerFunc("Despline16", "src:clip;width:int;height:int;src_left:float:opt;src_top:float:opt", despline16_create, nullptr, plugin);
}