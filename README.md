# video2vr

## What does it do?

Easily convert videos into side-by-side VR videos.

## How does it work?

The program estimates depth maps for each frame using AI. \
Using these depth maps it shifts the pixels for the left and right eye independendly.\
Then we place those 2 frames side-by-side and get a VR-video.

## How to start the program?

```bash
git clone https://github.com/xXChampionsXx/video2vr
cd video2vr
pip3 install -r requirements.txt
python video2vr.py
```

## GPU support / CUDA support

Install pytorch with cuda support following the instructions on here: https://pytorch.org/get-started/locally/#start-locally
