# bench_affine_compare.py
import os, time, argparse
import numpy as np
import cv2

from affine_naive import affine_naive
from affine_neon import affine_neon


# -------------------- Utils --------------------
def draw_label(img, text, org, color=(255, 255, 255)):
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def random_affine_matrix(center):
    angle = np.random.uniform(-45, 45)
    scale = np.random.uniform(0.5, 1.5)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    tx = np.random.uniform(-0.2, 0.2) * center[0] * 2
    ty = np.random.uniform(-0.2, 0.2) * center[1] * 2
    M[0, 2] += tx
    M[1, 2] += ty
    return M.astype(np.float32), angle, scale, tx, ty


def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def time_once(fn, iters=1):
    for _ in range(3):  # warmup
        _ = fn()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return np.array(t), _


# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Affine warp benchmark (NEON vs Naive vs OpenCV)"
    )
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 3],
        default=3,
        help="Number of channels (1=grayscale, 3=RGB)",
    )
    parser.add_argument("--threads", type=int, default=2, help="OMP_NUM_THREADS")
    parser.add_argument(
        "--iters", type=int, default=20, help="Number of measurement iterations"
    )
    args = parser.parse_args()

    H, W, C = args.height, args.width, args.channels
    num_threads = args.threads
    iters = args.iters
    np.random.seed(123)

    # OMP settings
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

    # Input image
    if C == 1:
        src = np.random.randint(0, 256, (H, W), dtype=np.uint8)
        fill_value = np.array([0], dtype=np.uint8)
    else:
        src = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        fill_value = np.array([0, 0, 0], dtype=np.uint8)

    center = (W / 2.0, H / 2.0)
    M, angle, scale, tx, ty = random_affine_matrix(center)
    print(f"angle={angle:.2f}, scale={scale:.2f}, tx={tx:.1f}, ty={ty:.1f}")

    def run_simd():
        return affine_neon(src, M, (W, H), fill_value)

    def run_naive():
        return affine_naive(src, M, (W, H), fill_value)

    def run_cv2():
        if C == 1:
            border_val = (int(fill_value[0]),)
        else:
            border_val = tuple(int(v) for v in fill_value)
        return cv2.warpAffine(
            src,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )

    # Measure
    t_simd, out_simd = time_once(run_simd, iters=iters)
    t_naiv, out_naiv = time_once(run_naive, iters=iters)
    t_cv2, out_cv2 = time_once(run_cv2, iters=iters)

    def stats(name, arr):
        print(
            f"{name:>20s}: mean={arr.mean():6.3f} ms  median={np.median(arr):6.3f} ms  p95={np.percentile(arr, 95):6.3f} ms  min={arr.min():6.3f} ms"
        )

    print("\nLatency stats (lower is better)")
    stats("SIMD warp_affine", t_simd)
    stats("Naive (OpenMP)", t_naiv)
    stats("OpenCV warpAffine", t_cv2)

    # Accuracy check
    diff_simd = np.abs(out_simd.astype(np.int16) - out_cv2.astype(np.int16))
    diff_naiv = np.abs(out_naiv.astype(np.int16) - out_cv2.astype(np.int16))
    print("\nAccuracy vs OpenCV (BORDER_CONSTANT=0)")
    print(
        f"SIMD:  max={diff_simd.max()}  mean={diff_simd.mean():.3f}  PSNR={psnr(out_simd, out_cv2):.2f} dB"
    )
    print(
        f"Naive: max={diff_naiv.max()}  mean={diff_naiv.mean():.3f}  PSNR={psnr(out_naiv, out_cv2):.2f} dB"
    )

    # Visualization (only if C == 3)
    if C == 3:
        vis_row1 = np.concatenate([out_simd, out_cv2, out_naiv], axis=1)
        l1_simd = diff_simd.mean(axis=2).astype(np.uint8)
        l1_naiv = diff_naiv.mean(axis=2).astype(np.uint8)
        heat_simd = cv2.applyColorMap(l1_simd, cv2.COLORMAP_JET)
        heat_naiv = cv2.applyColorMap(l1_naiv, cv2.COLORMAP_JET)
        vis_row2 = np.concatenate(
            [heat_simd, np.zeros_like(heat_simd), heat_naiv], axis=1
        )

        vis = np.concatenate([vis_row1, vis_row2], axis=0)
        w = out_simd.shape[1]
        draw_label(vis, "SIMD (custom)", (30, 50), (0, 255, 255))
        draw_label(vis, "OpenCV", (w + 30, 50), (0, 255, 0))
        draw_label(vis, "Naive", (2 * w + 30, 50), (255, 200, 0))
        draw_label(vis, "Diff vs OpenCV", (30, out_simd.shape[0] + 50), (0, 255, 255))
        draw_label(vis, " ", (w + 30, out_simd.shape[0] + 50), (0, 0, 0))
        draw_label(
            vis, "Diff vs OpenCV", (2 * w + 30, out_simd.shape[0] + 50), (255, 200, 0)
        )

        out_file = f"affine_comparison_random_{W}x{H}_c{C}.png"
        cv2.imwrite(out_file, vis)
        print(f"\nSaved: {out_file}")
