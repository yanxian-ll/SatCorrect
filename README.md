# SatCorrect

## Following

- [https://github.com/Kai-46/SatelliteSfM](https://github.com/Kai-46/SatelliteSfM)
- [https://github.com/Kai-46/ColmapForVisSat](https://github.com/Kai-46/ColmapForVisSat)
- [https://github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
- [https://github.com/pubgeo/dfc2019/tree/master/track3](https://github.com/pubgeo/dfc2019/tree/master/track3)


## How to use

```bash
python satellite_sfm.py --scene_path scene

```

## Input Structure

```
scene/
├── input/
│   ├── IMG_0.tif
│   ├── IMG_1.tif
│   │   ├── ...
├── scene_DSM.tif
├── scene_DSM.txt
...


## Output Structure

```
scene/
├── images/...
├── sparse/
│   ├── base/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   ├── points3D.bin
│   ├── 0/...
...

