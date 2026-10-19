# CTR-GCN (vendored)

`ctrgcn.py`, `ntu_rgb_d.py`, `tools.py` (graph/tools.py) and `feeder_tools.py` (feeders/tools.py) are copied unmodified from the official implementation
https://github.com/Uason-Chen/CTR-GCN (commit e13d7582e281d06711eeecb380a472b278ae1663), except:

- `ntu_rgb_d.py`: `sys.path.extend(['../'])` + `from graph import tools` replaced by `from . import tools`
  so the package imports from this location. No change to the model or graph.

Chen et al., "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action
Recognition", ICCV 2021. Licensed CC BY-NC 4.0 (see `LICENSE`): non-commercial use with attribution.
