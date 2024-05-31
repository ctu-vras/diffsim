# Differentiable Simulation

Differentiable robot-terrain interaction model for a tracked robot written
using [NVIDIA-Warp](https://nvidia.github.io/warp/).

![](./docs/intro.png)

## Running

To train Terrain Encoder model with the L2-loss
computed between predicted (by Differentiable Physics)
and ground truth trajectories, run:

```bash
python terrain_optimization
```
