# time-lapser

Remove long, visually static stretches from timelapse LEGO build videos. Time-lapser
samples the video, detects changes between frames, and keeps the activity with a
small amount of surrounding footage.

## Requirements

- Python 3
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on your `PATH`)

## Usage

```sh
python3 time_lapser.py input.mp4 output.mp4
```

By default, frames are sampled every second. Changes less than 8 seconds apart
are treated as one active stretch, with 1.5 seconds of padding retained around
each stretch. Tune detection and trimming with:

```sh
python3 time_lapser.py input.mp4 output.mp4 \
  --sample-interval 1 \
  --threshold 2 \
  --min-static-duration 8 \
  --padding 1.5
```

`--threshold` is the minimum average grayscale pixel difference (0–255) between
sampled frames. Lower it to detect smaller changes; raise it to ignore minor
lighting or camera noise. If no activity is detected, the tool leaves the
source untouched and does not create an output file. Existing output files are
never overwritten.

Run the unit tests with `python3 -m unittest discover -s tests`.
