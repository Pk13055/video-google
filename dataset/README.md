## Dataset

We have used Friends S01E01 for the `tf_table` and shot features. You can
_potentially_ use another video with the same scripts but there might be a
couple of changes.

- The video was cropped to a 10:00 duration (including starting credits).
- The frame resolution is 496x384
- Using a naive `ffmpeg` shot detection, we were able to collet around 190
  frames. The command used for the same is:
```
ffmpeg -i raw.mp4 -vf select='gt(scene\,0.2)' -vsync vfr %d.png 
```

