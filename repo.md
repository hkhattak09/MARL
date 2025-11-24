**Overview**
- **Purpose**: LAMARL (LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation) - a research framework combining Large Language Models with Multi-Agent RL for cooperative swarm control
- **Paper**: "LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation" published in IEEE Robotics and Automation Letters (2025)
- **Main Components**: 
  - LLM-based automatic function generation for reward shaping and policy guidance
  - MADDPG-based multi-agent training with cooperative policies
  - Custom Gym environment for assembly swarm tasks
  - C++ accelerated physics simulation for efficient multi-agent interactions
  - Comprehensive evaluation and visualization tools
- **Key Innovation**: Uses LLMs to generate prior actions/behaviors that guide RL policy learning via regularization, accelerating convergence and improving coordination

**Repository Structure**
- **`marl_llm/`**: Main project for MARL + LLM modules.
  - **`cfg/`**: Configuration files (e.g., `assembly_cfg.py`).
  - **`train/`**: Training scripts (e.g., `train_assembly.py`).
  - **`eval/`**: Evaluation scripts (e.g., `eval_assembly.py`).
  - **`llm/`**: LLM modules and framework for function generation (API connectors and helper scripts).
  - **`algorithm/`**: RL algorithms, replay buffers, agents, and utilities.

- **`cus_gym/`**: Custom Gym environment adapted/extended from OpenAI Gym.
  - **`gym/`**: Environment implementations and wrappers.
  - **`gym/envs/customized_envs/`**: Domain-specific environments (C++ helpers live in `envs_cplus`).
  - **`setup.py`**: Local package install script (install `cus_gym` in editable mode).

- **`cfg/`** (top-level): Additional configuration files referenced by training scripts.
- **`fig/`**, **`train/`**, **`eval/`**, **`test/`**: Experiment data, saved models, logs, and recorded runs.

**Key Components & How They Work**

1. **Custom Gym Environment (`cus_gym/gym`)**
   - Based on OpenAI Gym API with custom extensions for multi-agent swarm environments
   - Core classes in `cus_gym/gym/core.py`:
     - `Env`: Base environment with `step()`, `reset()`, `render()`, `seed()`, `close()` methods
     - `Wrapper`: Modular environment transformation class
     - `GoalEnv`: Goal-based environment for reward computation
     - Wrapper types: `ObservationWrapper`, `RewardWrapper`, `ActionWrapper`
   
2. **Assembly Swarm Environment** (`cus_gym/gym/envs/customized_envs/assembly.py`)
   - Multi-agent cooperative assembly task with configurable grid targets
   - Key features:
     - Processes target shape images (PNG) and extracts grid coordinates for agents to occupy
     - Configurable boundaries (wall or periodic), agent dynamics (Cartesian), collision detection
     - Metrics: coverage rate, distribution uniformity, Voronoi-based uniformity
     - C++ acceleration for physics calculations (contact forces, collision detection)
   - Environment wrapper: `AssemblySwarmWrapper` provides multi-agent interface with metrics

3. **C++ Acceleration** (`cus_gym/gym/envs/customized_envs/envs_cplus/`)
   - Performance-critical physics computations implemented in C++
   - Loaded via `ctypes` (see `c_lib.py` for platform-specific library loading)
   - Build script: `build.sh` (compiles to `.so` on Linux, `.dylib` on macOS, `.dll` on Windows)
   - Provides functions for inter-agent forces, wall collisions, and spatial queries

4. **MADDPG Algorithm** (`marl_llm/algorithm/algorithms/maddpg.py`)
   - Multi-Agent Deep Deterministic Policy Gradient implementation
   - Architecture:
     - Each agent has: policy network, critic network, and their target networks
     - Networks: 4-layer MLP with LeakyReLU activation (configurable hidden dim)
     - Actor-critic update with optional regularization towards prior actions (for LLM guidance)
   - Key methods:
     - `init_from_env()`: Factory method to create MADDPG from environment specs
     - `step()`: Sample actions from policies with optional exploration noise
     - `update()`: Perform actor-critic update using TD error and policy gradient
     - `update_all_targets()`: Soft update of target networks
     - `prep_training()` / `prep_rollouts()`: Switch between training and evaluation modes
   - Supports Gaussian exploration noise with epsilon-greedy random actions

5. **Replay Buffer** (`marl_llm/algorithm/utils/buffer_agent.py`)
   - `ReplayBufferAgent`: Circular buffer storing (s, a, r, s', done) tuples for all agents
   - Supports prior actions (for LLM-guided learning) and log probabilities
   - Efficient batch sampling with GPU transfer support
   - Handles multi-agent indexing via start/stop slices

6. **Training Loop** (`marl_llm/train/train_assembly.py`)
   - Main training flow:
     1. Setup logging (TensorBoard), model directories, random seeds
     2. Initialize environment and MADDPG agents
     3. For each episode:
        - Collect rollout: agents interact with environment, store transitions in replay buffer
        - Training phase: sample batches and perform multiple gradient updates
        - Soft update target networks
        - Log metrics and save checkpoints
   - Features:
     - Live rendering with matplotlib (updates every N frames)
     - Multi-device support: CPU, CUDA GPU, Apple Silicon MPS
     - Configurable batch sizes (default 2048) and training iterations (20-30 per episode)
     - Noise annealing: exploration noise decays over episodes

7. **Configuration System** (`marl_llm/cfg/assembly_cfg.py`)
   - Image preprocessing: loads target shape images, extracts grid coordinates, scales/rotates targets
   - Saves processed shapes to `fig/results.pkl` for fast loading during training
   - Environment parameters:
     - `n_a`: number of agents (default 30)
     - `boundary_width_half`, `boundary_height_half`: arena size (default 2.4×2.4)
     - `r_avoid`: collision avoidance radius (computed from agent count and target size)
     - `topo_nei_max`: max topological neighbors per agent (default 6)
   - Training hyperparameters:
     - Learning rates: `lr_actor=1e-4`, `lr_critic=1e-3`
     - Batch size: 2048, Buffer size: 50k
     - Hidden dimension: 256, Episodes: 3000, Episode length: 200
     - Exploration: `epsilon=0.1`, `noise_scale=0.9`, `tau=0.01` (soft update)
   
8. **LLM Integration** (`marl_llm/llm/`)
   - Purpose: Generate reward functions or behavior policies using LLMs
   - Structure:
     - `modules/framework/actions/rl_generate_functions.py`: Main entry point for function generation
     - `modules/llm/`: LLM API connectors (GPT, Claude, Qwen, VLM)
     - `config/llm_config.yaml`: API keys and model configurations
   - Workflow (from `RLGenerateFunctions`):
     - `RLGeneration`: Generate candidate functions
     - `RLCodeReview`: Review and validate generated code
     - Supports concurrent processing with asyncio (up to 30 parallel requests)
   - Training method: `llm_rl` mode uses generated prior actions for policy regularization

9. **Evaluation** (`marl_llm/eval/eval_assembly.py`)
   - Load trained models via `MADDPG.init_from_save()`
   - Process multiple shapes with controlled transformations (rotation, offset)
   - Compute swarm metrics: coverage, uniformity, success rate
   - Generate result files in `models/<env_name>/<timestamp>/results/`

10. **Network Architectures** (`marl_llm/algorithm/utils/networks.py`)
    - `MLPNetwork`: 4-layer fully connected network with LeakyReLU
      - Input → FC(hidden) → FC(hidden) → FC(hidden) → FC(output)
      - Optional tanh output constraint for continuous actions
    - `MLPNetworkRew`: MLP with residual blocks for reward learning
    - `Discriminator`: Used in AIRL (Adversarial IRL) for reward inference

**Setup & Installation**
- **Python**: Project is tested with Python 3.10 (see top-level `README.md`).
- **Environment**: Recommended process:

```bash
conda create -n marl_llm python=3.10
conda activate marl_llm
```

- **Install MARL requirements**:

```bash
cd marl_llm
pip install -r requirements.txt
```

- **Install PyTorch** (GPU example shown in README):

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

- **Install and register custom Gym environment**:

```bash
cd cus_gym
pip install -e .
```

- **Set PYTHONPATH** (or add to shell rc):

```bash
echo 'export PYTHONPATH="$PYTHONPATH:/path/to/your/marl_llm/"' >> ~/.bashrc
source ~/.bashrc
```

- **Compile C++ components** (if using the C++ accelerated parts):

```bash
cd cus_gym/gym/envs/customized_envs/envs_cplus
chmod +x build.sh
./build.sh
```

**Running Training & Evaluation**
- **Training (example)**: configure `cfg/assembly_cfg.py` (set `image_folder`, seeds, hyperparams) then run:

```bash
cd marl_llm/train
python train_assembly.py
```

- **Evaluation**: update experiment run id in `marl_llm/eval/eval_assembly.py` (`curr_run`) and run:

```bash
python eval_assembly.py
```

- **LLM-based generation** (example script referenced in README):

```bash
python ./marl_llm/llm/modules/framework/actions/rl_generate_functions.py
```

**Configuration**
- **`cfg/assembly_cfg.py`**: Main config used by `train_assembly.py`. Contains:
  - **Image Processing**: Loads PNG files from `fig/`, extracts grid coordinates, applies transformations
  - **Environment Settings**:
    - Agent count (`n_a=30`), boundary mode (`is_boundary=True`), dynamics (`dynamics_mode='Cartesian'`)
    - Observation/action spaces, collision radius, sensor range
    - Live rendering (`live_render=True`)
  - **Training Hyperparameters**:
    - Seeds (`seed=226`), device (`device='gpu'`), threads
    - Buffer capacity (`buffer_length=5e4`), episodes (`n_episodes=3000`), episode length (`episode_length=200`)
    - Batch size (`batch_size=2048`), hidden dim (`hidden_dim=256`)
    - Learning rates: actor `1e-4`, critic `1e-3`
    - Exploration: epsilon `0.1`, noise scale `0.9`, tau `0.01`
    - Training method: `llm_rl` (LLM-guided) or `manual_rl` (standard MADDPG)
  - **IRL Parameters**: discriminator learning rate, hidden layers (for AIRL algorithm)

**Developer Notes**

**Code Organization:**
- **Entry Points:**
  - Training: `marl_llm/train/train_assembly.py` (main training loop)
  - Evaluation: `marl_llm/eval/eval_assembly.py` (evaluate trained models)
  - LLM generation: `marl_llm/llm/modules/framework/actions/rl_generate_functions.py`
  - Config: `marl_llm/cfg/assembly_cfg.py` (hyperparameters and environment setup)

- **Algorithm Stack:**
  - `marl_llm/algorithm/algorithms/`: RL algorithms (MADDPG, AIRL)
  - `marl_llm/algorithm/utils/`: Supporting utilities
    - `agents.py`: DDPGAgent class (actor-critic agent)
    - `buffer_agent.py`: Replay buffer implementation
    - `networks.py`: Neural network architectures (MLP, discriminator)
    - `noise.py`: Exploration noise (Gaussian, OU noise)
    - `misc.py`: Helper functions (soft update, gradient averaging)

- **Environment Stack:**
  - `cus_gym/gym/core.py`: Base Gym API classes
  - `cus_gym/gym/envs/customized_envs/assembly.py`: Assembly environment (978 lines)
  - `cus_gym/gym/wrappers/customized_envs/assembly_wrapper.py`: Multi-agent wrapper with metrics
  - `cus_gym/gym/envs/customized_envs/envs_cplus/`: C++ acceleration modules

**Key Algorithms:**

1. **MADDPG Training Flow:**
   ```
   For each episode:
     1. Reset environment, sample target shape
     2. Rollout phase (CPU):
        - For each timestep:
          - Get actions from policies (with exploration noise)
          - Step environment, store (s,a,r,s',done) in buffer
     3. Training phase (GPU):
        - For 20-30 iterations:
          - Sample batch from buffer
          - Update critic: minimize TD error
          - Update actor: maximize Q-value + optional LLM regularization
        - Soft update target networks
     4. Log metrics, save checkpoints
   ```

2. **Actor-Critic Update:**
   - **Critic loss**: MSE between Q(s,a) and r + γ·Q_target(s', a')
   - **Actor loss**: -Q(s, π(s)) + α·MSE(π(s), prior_action) [if LLM guidance enabled]
   - Target networks updated via soft update: θ_target ← τ·θ + (1-τ)·θ_target

3. **Environment Reset (Domain Generalization):**
   - Randomly select shape from multiple training shapes
   - Apply random scale (1.0 to 1.3)
   - Apply random rotation (-π to π)
   - Apply random position offset
   - Agents start at random positions

**Testing & Debugging:**
- Unit tests: `cus_gym/spaces/tests/`, `cus_gym/utils/tests/`
- Run with `pytest` from repository root
- TensorBoard logs: `marl_llm/train/models/<env_name>/<timestamp>/logs/`
  - View with: `tensorboard --logdir=marl_llm/train/models/<env_name>/<timestamp>/logs`

**Model Checkpoints:**
- Saved to: `marl_llm/train/models/<env_name>/<timestamp>/`
- Final model: `model.pt`
- Incremental: `incremental/model_ep{N}.pt` (saved every 4×save_interval episodes)

**Performance Tips:**
- Increase `batch_size` (default 2048) for better GPU utilization
- Adjust `hidden_dim` (default 256) for model capacity vs speed tradeoff
- Use `device='gpu'` for faster training (10-30x speedup vs CPU)
- C++ acceleration critical for large swarms (30+ agents)
- Live rendering slows training; disable for production runs

**Troubleshooting & Tips**

**Common Issues:**
- **Missing packages**: Install via `pip install <package>` or add to `requirements.txt`
- **CUDA/PyTorch issues**: 
  - Verify GPU compatibility: `nvidia-smi`
  - Check CUDA version matches PyTorch wheel
  - Install correct wheel from [PyTorch website](https://pytorch.org/get-started/)
- **C++ compilation errors**: 
  - Ensure C++ compiler installed (GCC on Linux, Clang on macOS, MinGW on Windows)
  - Install build tools: `sudo apt-get install build-essential` (Ubuntu)
  - Check compiler version: `g++ --version` (need GCC 7+ or Clang 5+)
- **Import errors**: 
  - Verify `PYTHONPATH` includes `marl_llm/` directory
  - Check `cus_gym` installed with `pip list | grep gym`
- **Out of memory (GPU)**: 
  - Reduce `batch_size` (try 1024 or 512)
  - Reduce `hidden_dim` (try 128 or 64)
  - Reduce number of agents `n_a` during development

**Dependencies:**
- **Python**: 3.10 (tested and recommended)
- **PyTorch**: 2.1.0 with CUDA 12.1 (or compatible version)
- **Key packages** (from `marl_llm/requirements.txt`):
  - `numpy>=1.21`, `scipy<1.14`, `matplotlib>=3.5`
  - `opencv-python>=4.5`, `pillow>=10.2`
  - `tensorboardX>=2.6` (for logging)
  - `cloudpickle>=3.0`, `tqdm>=4.66` (utilities)
  - `networkx>=3.3`, `sympy>=1.12` (graph/symbolic math)
- **cus_gym dependencies** (from `cus_gym/requirements.txt`):
  - `atari-py==0.2.6`, `box2d-py~=2.3.5`, `pyglet>=1.4.0`
  - `mujoco_py>=1.50` (optional, for MuJoCo environments)

**LLM API Setup:**
- Edit `marl_llm/llm/config/llm_config.yaml` with your API keys
- Supported models: GPT-4, Claude, Qwen, VLM
- Usage: `python ./marl_llm/llm/modules/framework/actions/rl_generate_functions.py`

**Platform-Specific Notes:**
- **macOS**: Use MPS device for Apple Silicon GPUs (`device='mps'`)
- **Windows**: Ensure MinGW/MSYS2 installed for C++ compilation
- **Linux**: Most tested platform, recommended for production

**Useful Files & Entry Points**
- **Training**: `marl_llm/train/train_assembly.py` (main training script)
- **Evaluation**: `marl_llm/eval/eval_assembly.py` (evaluate trained policies)
- **Configuration**: `marl_llm/cfg/assembly_cfg.py` (hyperparameters, environment settings)
- **LLM Generation**: `marl_llm/llm/modules/framework/actions/rl_generate_functions.py`
- **Environment Core**: 
  - `cus_gym/gym/core.py` (base Gym API)
  - `cus_gym/gym/envs/customized_envs/assembly.py` (assembly environment, 978 lines)
  - `cus_gym/gym/wrappers/customized_envs/assembly_wrapper.py` (multi-agent wrapper)
- **Algorithms**:
  - `marl_llm/algorithm/algorithms/maddpg.py` (MADDPG implementation, 308 lines)
  - `marl_llm/algorithm/algorithms/airl.py` (Adversarial IRL)
  - `marl_llm/algorithm/utils/agents.py` (DDPGAgent class)
  - `marl_llm/algorithm/utils/networks.py` (neural network architectures)
- **C++ Acceleration**: `cus_gym/gym/envs/customized_envs/envs_cplus/` (build.sh, c_lib.py)

**Additional Resources:**
- **Documentation**: `cus_gym/docs/` (API docs, tutorials, environment guides)
- **Examples**: See `marl_llm/train/` and `marl_llm/eval/` for usage patterns
- **Logs**: TensorBoard logs in `marl_llm/train/models/<env_name>/<timestamp>/logs/`
- **Models**: Saved checkpoints in `marl_llm/train/models/<env_name>/<timestamp>/`
- **Results**: Evaluation outputs in `marl_llm/train/models/<env_name>/<timestamp>/results/`

**Citation & License**
- Cite the paper (see `README.md`) if you use this code. Project is MIT licensed (see `LICENSE.md`).

---

**Summary**

This repository implements LAMARL, a framework that combines Large Language Models with Multi-Agent Reinforcement Learning for cooperative swarm control. The system uses LLMs to generate prior behaviors that guide MADDPG training, enabling faster convergence and better coordination in multi-agent assembly tasks.

**Architecture Overview:**
- **Environment**: Custom OpenAI Gym environment with C++ acceleration for 30+ agent swarms performing cooperative assembly
- **Algorithm**: MADDPG (Multi-Agent DDPG) with optional LLM-guided regularization
- **LLM Integration**: Generates reward functions and prior actions to guide policy learning
- **Training**: Actor-critic updates with experience replay, soft target updates, and exploration noise
- **Evaluation**: Metrics for coverage, uniformity, and task success

For questions or issues, refer to the [GitHub repository](https://github.com/Guobin-Zhu/MARL-LLM) or cite the paper if using this code in your research.

---

*This documentation was generated by reviewing the codebase on November 24, 2025. For the latest updates, check the repository.*
