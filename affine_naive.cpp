#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using u8 = unsigned char;

static inline u8 sat_cast_u8(float v) noexcept
{
    int vi = int(v + 0.5f);
    if (vi < 0) vi = 0;
    if (vi > 255) vi = 255;
    return (u8)vi;
}

static inline void invertAffine2x3(const float M[6], float invM[6])
{
    const float a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5];
    const float det = a * e - b * d;
    if (std::fabs(det) < 1e-12f)
        throw std::runtime_error("Singular matrix");
    const float ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
    invM[0] = ia;  invM[1] = ib;  invM[2] = -(ia * c + ib * f);
    invM[3] = id;  invM[4] = ie;  invM[5] = -(id * c + ie * f);
}

static inline void init_src_coords_row(int y, float ib_, float ic, float ie, float if_, float &sx, float &sy) noexcept
{
    sx = ib_ * y + ic;
    sy = ie * y + if_;
}

static inline void step_src_coords(float &sx, float &sy, float dsx, float dsy) noexcept
{
    sx += dsx;  sy += dsy;
}

template <int C>
static inline void interpolate_store(const u8 *src, int W, int H,
                                     float sx, float sy,
                                     u8 *dst_px, const u8 *fill_value) noexcept
{
    int   x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
    float fx = sx - x0,          fy = sy - y0;
    int   x1 = x0 + 1,           y1 = y0 + 1;

    auto inside = [&](int x, int y) -> bool {
        return (unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H;
    };

    if constexpr (C == 1)
    {
        const u8 p00 = inside(x0, y0) ? src[(size_t)y0 * W + x0] : fill_value[0];
        const u8 p01 = inside(x1, y0) ? src[(size_t)y0 * W + x1] : fill_value[0];
        const u8 p10 = inside(x0, y1) ? src[(size_t)y1 * W + x0] : fill_value[0];
        const u8 p11 = inside(x1, y1) ? src[(size_t)y1 * W + x1] : fill_value[0];

        float h0 = p00 + (p01 - p00) * fx;
        float h1 = p10 + (p11 - p10) * fx;
        dst_px[0] = sat_cast_u8(h0 + (h1 - h0) * fy);
    }
    else // C == 3
    {
        // y が範囲内のときだけ行ポインタを有効に
        const u8 *r0 = ((unsigned)y0 < (unsigned)H) ? src + (size_t)y0 * W * 3 : nullptr;
        const u8 *r1 = ((unsigned)y1 < (unsigned)H) ? src + (size_t)y1 * W * 3 : nullptr;

        u8 q00[3], q01[3], q10[3], q11[3];
        for (int c = 0; c < 3; ++c) {
            q00[c] = (r0 && (unsigned)x0 < (unsigned)W) ? r0[x0 * 3 + c] : fill_value[c];
            q01[c] = (r0 && (unsigned)x1 < (unsigned)W) ? r0[x1 * 3 + c] : fill_value[c];
            q10[c] = (r1 && (unsigned)x0 < (unsigned)W) ? r1[x0 * 3 + c] : fill_value[c];
            q11[c] = (r1 && (unsigned)x1 < (unsigned)W) ? r1[x1 * 3 + c] : fill_value[c];
        }
        for (int c = 0; c < 3; ++c) {
            float h0 = q00[c] + (q01[c] - q00[c]) * fx;
            float h1 = q10[c] + (q11[c] - q10[c]) * fx;
            dst_px[c] = sat_cast_u8(h0 + (h1 - h0) * fy);
        }
    }
}

template <int C, bool kUseRowIncrement>
static inline void process_row(const u8 *src, int W, int H,
                               int y, int Wout,
                               float ia, float ib_, float ic,
                               float id, float ie, float if_,
                               const u8 *fill_value,
                               u8 *drow) noexcept
{
    if constexpr (kUseRowIncrement) {
        float sx, sy;
        init_src_coords_row(y, ib_, ic, ie, if_, sx, sy);
        for (int x = 0; x < Wout; ++x) {
            interpolate_store<C>(src, W, H, sx, sy, drow + x * C, fill_value);
            step_src_coords(sx, sy, ia, id);
        }
    } else {
        for (int x = 0; x < Wout; ++x) {
            float sx = ia * x + ib_ * y + ic;
            float sy = id * x + ie * y + if_;
            interpolate_store<C>(src, W, H, sx, sy, drow + x * C, fill_value);
        }
    }
}

py::array_t<u8> affine_naive(
    py::array_t<u8, py::array::c_style | py::array::forcecast> src_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> M_in,
    std::pair<int, int> out_wh,
    py::array_t<u8, py::array::c_style | py::array::forcecast> fill_value_in)
{
    auto ib = src_in.request();
    if (ib.ndim != 2 && ib.ndim != 3)
        throw std::runtime_error("src must be HxW or HxWx3");

    const int H = (int)ib.shape[0], W = (int)ib.shape[1];
    const int C = (ib.ndim == 2) ? 1 : (int)ib.shape[2];
    if (H <= 0 || W <= 0) throw std::runtime_error("empty input");
    if (C != 1 && C != 3) throw std::runtime_error("channels must be 1 or 3");

    auto fb = fill_value_in.request();
    if (fb.ndim != 1 || fb.shape[0] != C)
        throw std::runtime_error("fill_value must match number of channels");
    const u8 *fill_value = static_cast<const u8 *>(fb.ptr);

    auto mb = M_in.request();
    if (mb.ndim != 2 || mb.shape[0] != 2 || mb.shape[1] != 3)
        throw std::runtime_error("M must be 2x3");

    float Minv[6];
    invertAffine2x3(static_cast<const float *>(mb.ptr), Minv);
    const float ia = Minv[0], ib_ = Minv[1], ic = Minv[2];
    const float id = Minv[3], ie = Minv[4], if_ = Minv[5];

    const int Wout = out_wh.first, Hout = out_wh.second;
    if (Wout <= 0 || Hout <= 0) throw std::runtime_error("Invalid output size");

    py::array_t<u8, py::array::c_style> out =
        (C == 1) ? py::array_t<u8, py::array::c_style>({Hout, Wout})
                 : py::array_t<u8, py::array::c_style>({Hout, Wout, C});

    auto ob = out.request();
    const u8 *src = static_cast<const u8 *>(ib.ptr);
    u8 *dst = static_cast<u8 *>(ob.ptr);

    constexpr bool kUseRowIncrement = true;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < Hout; ++y) {
        u8 *drow = dst + (size_t)y * Wout * C;
        if (C == 1)
            process_row<1, kUseRowIncrement>(src, W, H, y, Wout, ia, ib_, ic, id, ie, if_, fill_value, drow);
        else
            process_row<3, kUseRowIncrement>(src, W, H, y, Wout, ia, ib_, ic, id, ie, if_, fill_value, drow);
    }
    return out;
}

PYBIND11_MODULE(affine_naive, m)
{
    m.doc() = "warpAffine bilinear with constant border value";
    m.def("affine_naive", &affine_naive,
          py::arg("src"), py::arg("M"), py::arg("out_wh"), py::arg("fill_value"));
}
