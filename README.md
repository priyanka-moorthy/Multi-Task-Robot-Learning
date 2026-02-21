# Multi-Task Vision-Language-Action Robot Learning

Training a single policy to perform multiple manipulation tasks using vision + language instructions.

## Project Structure

```
MultiTaskAgent/
├── src/
│   └── environments/
│       ├── base.py              # Abstract simulator interface
│       └── pybullet_env.py      # PyBullet implementation
├── data/                        # Training datasets (generated)
├── configs/                     # Model configurations
├── experiments/                 # Training runs and checkpoints
├── notebooks/                   # Jupyter notebooks for analysis
├── test_environment.py          # Test script
└── requirements.txt
```


## Setup

**Quick Start:**
```bash
cd /Users/priyanka/Documents/MultiTaskAgent
./install.sh
```

**Detailed instructions:** See [SETUP.md](SETUP.md)

The installer automatically detects your platform and installs the appropriate simulator:
- **Apple Silicon (M1/M2/M3)**: MuJoCo (recommended)
- **x86_64**: Choice of PyBullet or MuJoCo

## Test Environment

**Headless mode** (no GUI, saves images):
```bash
python test_environment.py --mode headless
```

**GUI mode** (watch robot in 3D viewer):
```bash
python test_environment.py --mode gui
```

## Available Tasks

1. **Pick**: "pick up the [red|blue|green] block"
2. **Push**: "push the [red|blue|green] block forward"
3. **Place**: "place the [red|blue|green] block on the target"

## Isaac Sim Migration Path

The code is designed to be simulator-agnostic. To migrate:

1. Implement `IsaacSimEnv(BaseRobotEnv)` following the same interface
2. Replace `FrankaPickAndPlaceEnv` with `IsaacSimEnv` in training scripts
3. No changes needed to VLA model code!

See comments in [base.py](src/environments/base.py) for Isaac Sim implementation pattern.
