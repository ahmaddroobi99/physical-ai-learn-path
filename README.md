# Physical AI Lab — Python to Robotics Perception

A local learning path for the notebook in `notebooks/`. The same lessons run as a Python package: arrays, a synthetic scene, edges, feature matching, a PyTorch timing check, camera geometry, depth-to-3D, image motion, int8 quantization, and a small perception benchmark.

## Setup

The virtual environment directory is `learn/`. It is listed in `.gitignore`.

```bash
python3 -m venv learn
source learn/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` installs the CPU builds of PyTorch and TorchVision from the official CPU wheel index. This machine does not need a CUDA wheel for the lessons.

## Interactive lab

Open the teaching page. Each lesson explains one function, shows its arguments, and runs it when you move a control or press Run.

```bash
source learn/bin/activate
python -m physical_ai_lab.server
```

Then open http://127.0.0.1:8765

## Run the whole path in the terminal

```bash
source learn/bin/activate
python -m physical_ai_lab
```

That command writes `results/lab_report.txt` and the lesson figures under `results/`. Those outputs stay out of git.

The pretrained detector downloads weights on first use. Run it only when you want that lesson:

```bash
python -m physical_ai_lab --detect
```

## Tests

```bash
source learn/bin/activate
pytest
```

## Lessons

1. Normalize sensor readings and convert units.
2. Treat an image as a NumPy array and build a synthetic scene.
3. Time a Python loop against a vectorized reduction.
4. Detect Canny edges and Hough lines.
5. Match ORB keypoints between a frame and a shifted frame.
6. Time a PyTorch matrix multiply on CPU or CUDA.
7. Optionally run a pretrained Faster R-CNN detector.
8. Project a 3D point and back-project it with the camera intrinsics.
9. Turn a depth map into a point cloud.
10. Estimate image translation with phase correlation.
11. Quantize float32 weights to int8 and measure the error.
12. Run grayscale, edges, and ORB as one pipeline and benchmark it.

Settings shared by the lessons live in `configs/lab.json`.
