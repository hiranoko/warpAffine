# WarpAffine

## Build

```
$ g++ -O3 -fopenmp -march=armv8-a+simd -mcpu=cortex-a53 -shared -fPIC $(python3 -m pybind11 --includes) affine_naive.cpp -o affine_naive$(python3-config --extension-suffix) $(python3-config --ldflags)
```

```
$ g++ -O3 -fopenmp -march=armv8-a+simd -mcpu=cortex-a53 -shared -fPIC $(python3 -m pybind11 --includes) affine_neon.cpp -o affine_neon$(python3-config --extension-suffix) $(python3-config --ldflags)
```

## Environment

- KV260

## Result

```
(venv) sh-5.1$ python main.py --channels 3 --width 640 --height 512
angle=-42.15, scale=1.28, tx=-119.9, ty=97.6

Latency stats (lower is better)
    SIMD warp_affine: mean=16.069 ms  median=14.880 ms  p95=21.533 ms  min=14.439 ms
      Naive (OpenMP): mean= 9.100 ms  median= 8.976 ms  p95= 9.372 ms  min= 8.947 ms
   OpenCV warpAffine: mean=11.666 ms  median=11.660 ms  p95=11.709 ms  min=11.632 ms

Accuracy vs OpenCV (BORDER_CONSTANT=0)
SIMD:  max=7  mean=0.699  PSNR=47.58 dB
Naive: max=7  mean=0.712  PSNR=47.39 dB

Saved: affine_comparison_random_640x512_c3.png
```