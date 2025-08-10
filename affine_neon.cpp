#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace py = pybind11;
using u8 = unsigned char;

static inline u8 sat_cast_u8(float v) noexcept
{
    int vi = int(v + 0.5f);
    if (vi < 0)
        vi = 0;
    if (vi > 255)
        vi = 255;
    return (u8)vi;
}

static inline void invertAffine2x3(const float M[6], float invM[6])
{
    const float a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5];
    const float det = a * e - b * d;
    if (std::fabs(det) < 1e-12f)
        throw std::runtime_error("Singular matrix");
    const float ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
    invM[0] = ia;
    invM[1] = ib;
    invM[2] = -(ia * c + ib * f);
    invM[3] = id;
    invM[4] = ie;
    invM[5] = -(id * c + ie * f);
}

static inline void init_src_coords_row(int y, float ib_, float ic, float ie, float if_, float &sx, float &sy) noexcept
{
    sx = ib_ * y + ic;
    sy = ie * y + if_;
}

static inline void step_src_coords(float &sx, float &sy, float dsx, float dsy) noexcept
{
    sx += dsx;
    sy += dsy;
}

template <int C>
static inline void interpolate_store(const u8 *src, int W, int H,
                                     float sx, float sy,
                                     u8 *dst_px, const u8 *fill_value) noexcept
{
    int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
    float fx = sx - x0, fy = sy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;

    auto inside = [&](int x, int y) -> bool
    {
        // unsigned 比較で 0<=x<W, 0<=y<H を一発チェック
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
        const u8 *r0 = inside(0, y0) ? src + (size_t)y0 * W * 3 : nullptr;
        const u8 *r1 = inside(0, y1) ? src + (size_t)y1 * W * 3 : nullptr;

        u8 q00[3], q01[3], q10[3], q11[3];
        for (int c = 0; c < 3; ++c)
        {
            q00[c] = (inside(x0, y0) && r0) ? r0[x0 * 3 + c] : fill_value[c];
            q01[c] = (inside(x1, y0) && r0) ? r0[x1 * 3 + c] : fill_value[c];
            q10[c] = (inside(x0, y1) && r1) ? r1[x0 * 3 + c] : fill_value[c];
            q11[c] = (inside(x1, y1) && r1) ? r1[x1 * 3 + c] : fill_value[c];
        }
        for (int c = 0; c < 3; ++c)
        {
            float h0 = q00[c] + (q01[c] - q00[c]) * fx;
            float h1 = q10[c] + (q11[c] - q10[c]) * fx;
            dst_px[c] = sat_cast_u8(h0 + (h1 - h0) * fy);
        }
    }
}

#ifdef __ARM_NEON
static inline void interpolate_block4_neon_C1(
    const u8 *src, int W, int H,
    const int x0_[4], const int y0_[4], const float fx_[4], const float fy_[4],
    const u8 fill, u8 *dst4) noexcept
{
    auto inside = [&](int x, int y) -> bool
    {
        return (unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H;
    };

    float p00f[4], p01f[4], p10f[4], p11f[4];
    for (int i = 0; i < 4; ++i)
    {
        int x0 = x0_[i], y0 = y0_[i];
        int x1 = x0 + 1, y1 = y0 + 1;

        p00f[i] = inside(x0, y0) ? (float)src[(size_t)y0 * W + x0] : (float)fill;
        p01f[i] = inside(x1, y0) ? (float)src[(size_t)y0 * W + x1] : (float)fill;
        p10f[i] = inside(x0, y1) ? (float)src[(size_t)y1 * W + x0] : (float)fill;
        p11f[i] = inside(x1, y1) ? (float)src[(size_t)y1 * W + x1] : (float)fill;
    }

    float32x4_t fxv = vld1q_f32(fx_);
    float32x4_t fyv = vld1q_f32(fy_);
    float32x4_t p00 = vld1q_f32(p00f);
    float32x4_t p01 = vld1q_f32(p01f);
    float32x4_t p10 = vld1q_f32(p10f);
    float32x4_t p11 = vld1q_f32(p11f);

    float32x4_t h0 = vmlaq_f32(p00, vsubq_f32(p01, p00), fxv);
    float32x4_t h1 = vmlaq_f32(p10, vsubq_f32(p11, p10), fxv);
    float32x4_t val = vmlaq_f32(h0, vsubq_f32(h1, h0), fyv);

    val = vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(val, vdupq_n_f32(255.0f)));
    val = vaddq_f32(val, vdupq_n_f32(0.5f));
    int32x4_t vi = vcvtq_s32_f32(val);

    uint16x4_t u16 = vqmovun_s32(vi);
    uint16x8_t u16x8 = vcombine_u16(u16, vdup_n_u16(0));
    uint8x8_t u8x8 = vqmovn_u16(u16x8);
    vst1_lane_u32(reinterpret_cast<uint32_t *>(dst4), vreinterpret_u32_u8(u8x8), 0);
}

static inline void interpolate_block4_neon_C3(
    const u8 *src, int W, int H,
    const int x0_[4], const int y0_[4], const float fx_[4], const float fy_[4],
    const u8 fillRGB[3], u8 *dst12) noexcept
{
    auto inside = [&](int x, int y) -> bool
    {
        return (unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H;
    };

    float p00r[4], p01r[4], p10r[4], p11r[4];
    float p00g[4], p01g[4], p10g[4], p11g[4];
    float p00b[4], p01b[4], p10b[4], p11b[4];

    for (int i = 0; i < 4; ++i)
    {
        int x0 = x0_[i], y0 = y0_[i];
        int x1 = x0 + 1, y1 = y0 + 1;
        const u8 *r0 = inside(0, y0) ? src + (size_t)y0 * W * 3 : nullptr;
        const u8 *r1 = inside(0, y1) ? src + (size_t)y1 * W * 3 : nullptr;

        auto pick = [&](const u8 *row, int x, int c, u8 fillc) -> float
        {
            return (row && inside(x, (row == (src + (size_t)y0 * W * 3) ? y0 : y1)))
                       ? (float)row[x * 3 + c]
                       : (float)fillc;
        };

        p00r[i] = pick(r0, x0, 0, fillRGB[0]);
        p01r[i] = pick(r0, x1, 0, fillRGB[0]);
        p10r[i] = pick(r1, x0, 0, fillRGB[0]);
        p11r[i] = pick(r1, x1, 0, fillRGB[0]);

        p00g[i] = pick(r0, x0, 1, fillRGB[1]);
        p01g[i] = pick(r0, x1, 1, fillRGB[1]);
        p10g[i] = pick(r1, x0, 1, fillRGB[1]);
        p11g[i] = pick(r1, x1, 1, fillRGB[1]);

        p00b[i] = pick(r0, x0, 2, fillRGB[2]);
        p01b[i] = pick(r0, x1, 2, fillRGB[2]);
        p10b[i] = pick(r1, x0, 2, fillRGB[2]);
        p11b[i] = pick(r1, x1, 2, fillRGB[2]);
    }

    float32x4_t fxv = vld1q_f32(fx_);
    float32x4_t fyv = vld1q_f32(fy_);

    auto lerp2d = [&](float32x4_t p00, float32x4_t p01, float32x4_t p10, float32x4_t p11)
    {
        float32x4_t h0 = vmlaq_f32(p00, vsubq_f32(p01, p00), fxv);
        float32x4_t h1 = vmlaq_f32(p10, vsubq_f32(p11, p10), fxv);
        float32x4_t v = vmlaq_f32(h0, vsubq_f32(h1, h0), fyv);
        v = vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(v, vdupq_n_f32(255.0f)));
        v = vaddq_f32(v, vdupq_n_f32(0.5f));
        return vcvtq_s32_f32(v);
    };

    int32x4_t vr = lerp2d(vld1q_f32(p00r), vld1q_f32(p01r), vld1q_f32(p10r), vld1q_f32(p11r));
    int32x4_t vg = lerp2d(vld1q_f32(p00g), vld1q_f32(p01g), vld1q_f32(p10g), vld1q_f32(p11g));
    int32x4_t vb = lerp2d(vld1q_f32(p00b), vld1q_f32(p01b), vld1q_f32(p10b), vld1q_f32(p11b));

    for (int i = 0; i < 4; ++i)
    {
        dst12[i * 3 + 0] = (u8)vgetq_lane_s32(vr, i);
        dst12[i * 3 + 1] = (u8)vgetq_lane_s32(vg, i);
        dst12[i * 3 + 2] = (u8)vgetq_lane_s32(vb, i);
    }
}
#endif

template <int C, bool kUseRowIncrement>
static inline void process_row_scalar(const u8 *src, int W, int H,
                                      int y, int Wout,
                                      float ia, float ib_, float ic,
                                      float id, float ie, float if_,
                                      const u8 *fill_value,
                                      u8 *drow) noexcept
{
    if constexpr (kUseRowIncrement)
    {
        float sx, sy;
        init_src_coords_row(y, ib_, ic, ie, if_, sx, sy);
        for (int x = 0; x < Wout; ++x)
        {
            interpolate_store<C>(src, W, H, sx, sy, drow + x * C, fill_value);
            step_src_coords(sx, sy, ia, id);
        }
    }
    else
    {
        for (int x = 0; x < Wout; ++x)
        {
            float sx = ia * x + ib_ * y + ic;
            float sy = id * x + ie * y + if_;
            interpolate_store<C>(src, W, H, sx, sy, drow + x * C, fill_value);
        }
    }
}

#ifdef __ARM_NEON
// NEON化 row: kUseRowIncrement=true 前提
static inline void process_row_neon_C1(const u8 *src, int W, int H,
                                       int y, int Wout,
                                       float ia, float ib_, float ic,
                                       float id, float ie, float if_,
                                       const u8 *fill_value,
                                       u8 *drow) noexcept
{
    float sx0, sy0;
    init_src_coords_row(y, ib_, ic, ie, if_, sx0, sy0);

    const u8 fill = fill_value[0];

    int x = 0;
    for (; x + 3 < Wout; x += 4)
    {
        float sx[4] = {sx0 + ia * (x + 0), sx0 + ia * (x + 1), sx0 + ia * (x + 2), sx0 + ia * (x + 3)};
        float sy[4] = {sy0 + id * (x + 0), sy0 + id * (x + 1), sy0 + id * (x + 2), sy0 + id * (x + 3)};
        int x0[4], y0_[4];
        float fx[4], fy[4];
        for (int i = 0; i < 4; ++i)
        {
            x0[i] = (int)std::floor(sx[i]);
            y0_[i] = (int)std::floor(sy[i]);
            fx[i] = sx[i] - x0[i];
            fy[i] = sy[i] - y0_[i];
        }
        interpolate_block4_neon_C1(src, W, H, x0, y0_, fx, fy, fill, drow + x);
    }
    // 端数はスカラ
    for (; x < Wout; ++x)
    {
        float sx = sx0 + ia * x;
        float sy = sy0 + id * x;
        interpolate_store<1>(src, W, H, sx, sy, drow + x, &fill);
    }
}

static inline void process_row_neon_C3(const u8 *src, int W, int H,
                                       int y, int Wout,
                                       float ia, float ib_, float ic,
                                       float id, float ie, float if_,
                                       const u8 *fill_value,
                                       u8 *drow) noexcept
{
    float sx0, sy0;
    init_src_coords_row(y, ib_, ic, ie, if_, sx0, sy0);
    int x = 0;
    for (; x + 3 < Wout; x += 4)
    {
        float sx[4] = {sx0 + ia * (x + 0), sx0 + ia * (x + 1), sx0 + ia * (x + 2), sx0 + ia * (x + 3)};
        float sy[4] = {sy0 + id * (x + 0), sy0 + id * (x + 1), sy0 + id * (x + 2), sy0 + id * (x + 3)};
        int x0[4], y0_[4];
        float fx[4], fy[4];
        for (int i = 0; i < 4; ++i)
        {
            x0[i] = (int)std::floor(sx[i]);
            y0_[i] = (int)std::floor(sy[i]);
            fx[i] = sx[i] - x0[i];
            fy[i] = sy[i] - y0_[i];
        }
        interpolate_block4_neon_C3(src, W, H, x0, y0_, fx, fy, fill_value, drow + x * 3);
    }
    for (; x < Wout; ++x)
    {
        float sx = sx0 + ia * x;
        float sy = sy0 + id * x;
        interpolate_store<3>(src, W, H, sx, sy, drow + x * 3, fill_value);
    }
}
#endif // __ARM_NEON

py::array_t<u8> affine_neon(
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
    if (H <= 0 || W <= 0)
        throw std::runtime_error("empty input");
    if (C != 1 && C != 3)
        throw std::runtime_error("channels must be 1 or 3");

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
    if (Wout <= 0 || Hout <= 0)
        throw std::runtime_error("Invalid output size");

    py::array_t<u8, py::array::c_style> out = (C == 1)
                                                  ? py::array_t<u8, py::array::c_style>({Hout, Wout})
                                                  : py::array_t<u8, py::array::c_style>({Hout, Wout, C});

    auto ob = out.request();
    const u8 *src = static_cast<const u8 *>(ib.ptr);
    u8 *dst = static_cast<u8 *>(ob.ptr);

    constexpr bool kUseRowIncrement = true;

#pragma omp parallel for schedule(static)
    for (int y = 0; y < Hout; ++y)
    {
        u8 *drow = dst + (size_t)y * Wout * C;
#ifdef __ARM_NEON
        if constexpr (kUseRowIncrement)
        {
            if (C == 1)
                process_row_neon_C1(src, W, H, y, Wout, ia, ib_, ic, id, ie, if_, fill_value, drow);
            else
                process_row_neon_C3(src, W, H, y, Wout, ia, ib_, ic, id, ie, if_, fill_value, drow);
            continue;
        }
#endif
        // フォールバック（非NEON or kUseRowIncrement=false）
        if (C == 1)
            process_row_scalar<1, kUseRowIncrement>(src, W, H, y, Wout, ia, ib_, ic, id, ie, if_, fill_value, drow);
        else
            process_row_scalar<3, kUseRowIncrement>(src, W, H, y, Wout, ia, ib_, ic, id, ie, if_, fill_value, drow);
    }
    return out;
}

PYBIND11_MODULE(affine_neon, m)
{
    m.doc() = "warpAffine bilinear with constant border value (NEON-optimized)";
    m.def("affine_neon", &affine_neon,
          py::arg("src"), py::arg("M"), py::arg("out_wh"), py::arg("fill_value"));
}
