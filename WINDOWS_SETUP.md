# MARL-LLM - Complete Windows Setup Guide

## 📋 Prerequisites

Before you start, ensure you have:

1. **Windows 10/11** (64-bit)
2. **Python 3.10 or newer** (up to 3.13 supported)
   - Download from: https://www.python.org/downloads/
   - ⚠️ **Important**: During installation, check "Add Python to PATH"
3. **Git for Windows**
   - Download from: https://git-scm.com/download/win
4. **Visual Studio Build Tools** (for C++ compilation)
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++" workload
5. **CUDA Toolkit** (Optional, for GPU acceleration)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Only if you have an NVIDIA GPU
6. **CMake** (for C++ compilation)
   - Download from: https://cmake.org/download/
   - Or install via: `winget install Kitware.CMake`

---

## 🚀 Installation Steps

### Step 1: Clone the Repository

Open **PowerShell** or **Command Prompt** and run:

```powershell
cd C:\Users\YourUsername\Documents  # Or any directory you prefer
git clone https://github.com/YourUsername/MARL.git
cd MARL
```

### Step 2: Create Virtual Environment

Using **Python venv** (recommended):

```powershell
python -m venv marl_env
.\marl_env\Scripts\activate
```

**OR** using **Conda** (if you have Anaconda/Miniconda):

```powershell
conda create -n marl_llm python=3.10
conda activate marl_llm
```

You should see `(marl_env)` or `(marl_llm)` in your command prompt.

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch (with GPU support)

**For GPU (NVIDIA):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```powershell
pip install torch torchvision torchaudio
```

**Verify PyTorch installation:**
```powershell
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Step 5: Install Custom Gym Environment

Navigate to `cus_gym` folder and install:

```powershell
cd cus_gym
pip install -e .
```

This will install all core dependencies: numpy, scipy, matplotlib, pyglet, cloudpickle

**Optional environment dependencies** (if you need Atari/MuJoCo):
```powershell
pip install -e .[optional]
```

### Step 6: Install MARL-LLM Dependencies

Navigate to `marl_llm` folder and install:

```powershell
cd ..\marl_llm
pip install -r requirements.txt
```

This installs all necessary packages including:
- PyTorch (if not already installed)
- TensorboardX (for logging)
- OpenAI API client (for LLM integration)
- PyYAML, tqdm, and other utilities

### Step 7: Compile C++ Acceleration Library

The environment uses C++ for performance-critical operations.

**Navigate to the C++ source directory:**
```powershell
cd ..\cus_gym\gym\envs\customized_envs\envs_cplus
```

**Create build directory and compile:**
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**Verify compilation:**
```powershell
dir ..\*.dll  # Should show compiled library files
```

If compilation fails, see [Troubleshooting](#troubleshooting-c-compilation) below.

### Step 8: Set Up Environment Variables (Optional but Recommended)

Add the MARL project to Python path:

**PowerShell (temporary, for current session):**
```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;C:\Users\YourUsername\Documents\MARL\marl_llm"
```

**Permanent setup (System Environment Variables):**
1. Open "Environment Variables" (search in Start Menu)
2. Under "User variables", click "New"
3. Variable name: `PYTHONPATH`
4. Variable value: `C:\Users\YourUsername\Documents\MARL\marl_llm` (use your actual path)
5. Click OK

### Step 9: Prepare Training Data

1. **Create a figures directory** for target shapes:
```powershell
mkdir C:\Users\YourUsername\Documents\MARL\figures
```

2. **Add target shape images** (PNG/JPG format) to this directory
   - These images define the assembly target patterns
   - The system will process these into grid coordinates

3. **Update configuration** in `marl_llm\cfg\assembly_cfg.py`:
   - Open the file in a text editor
   - Find the line: `image_folder = 'your/path/to/figures/'`
   - Replace with your actual path: `image_folder = r'C:\Users\YourUsername\Documents\MARL\figures\'`
   - Note: Use raw string `r''` or double backslashes `\\` for Windows paths

---

## 🎮 Running Training

### Basic Training (No LLM)

1. **Navigate to training directory:**
```powershell
cd C:\Users\YourUsername\Documents\MARL\marl_llm\train
```

2. **Start training:**
```powershell
python train_assembly.py
```

**What to expect:**
- Training will start and display episode progress
- Models will be saved in `marl_llm/train/models/assembly/YYYY-MM-DD-HH-MM-SS/`
- TensorBoard logs will be saved in the same directory under `logs/`
- Training runs for 3000 episodes by default (~2-4 hours on GPU)

### Training with Live Visualization

Enable live rendering to watch the training in real-time:

1. **Edit `assembly_cfg.py`** (around line 186):
```python
parser.add_argument("--live_render", default=True, type=bool)  # Set to True
```

2. **Run training:**
```powershell
python train_assembly.py
```

A matplotlib window will show live agent movements during training.

### Advanced Training with Inverse RL (AIRL)

For expert demonstration learning:

```powershell
python train_assembly_airl.py
```

This requires expert data in `marl_llm/eval/expert_data/`.

---

## 📊 Monitoring Training

### Using TensorBoard

While training is running (or after), open a new terminal:

```powershell
cd C:\Users\YourUsername\Documents\MARL\marl_llm\train\models\assembly
tensorboard --logdir=.
```

Open your browser to: http://localhost:6006

You'll see:
- Episode rewards over time
- Training loss curves
- Other performance metrics

---

## 🧪 Evaluation

After training completes, evaluate the trained model:

1. **Find your experiment timestamp:**
   - Look in `marl_llm\train\models\assembly\`
   - You'll see a folder like `2025-11-24-14-30-25`

2. **Navigate to evaluation directory:**
```powershell
cd C:\Users\YourUsername\Documents\MARL\marl_llm\eval
```

3. **Edit `eval_assembly.py`:**
   - Open in text editor
   - Find line: `curr_run = '2025-01-19-15-58-03'`
   - Replace with your experiment timestamp: `curr_run = '2025-11-24-14-30-25'`

4. **Run evaluation:**
```powershell
python eval_assembly.py
```

This will:
- Load the trained model
- Run evaluation episodes
- Generate visualizations
- Display performance metrics (coverage rate, uniformity, etc.)

---

## 🤖 LLM Integration (Optional)

To use LLM for automatic reward function generation:

### Step 1: Configure LLM API Keys

1. **Edit `marl_llm/llm/config/llm_config.yaml`:**

```yaml
api_base:
  GPT: "https://api.openai.com/v1"
  QWEN: "your_qwen_api_endpoint"
  
api_key:
  GPT: "sk-your-openai-api-key-here"
  QWEN: "your-qwen-api-key"
  CLAUDE: "your-claude-api-key"
  
model:
  GPT: "gpt-4"
  QWEN: "qwen-vl-plus"
  CLAUDE: "claude-3-opus"
```

2. **Configure experiment settings** in `marl_llm/llm/config/experiment_config.yaml`

### Step 2: Generate Reward Functions

```powershell
cd C:\Users\YourUsername\Documents\MARL\marl_llm\llm\modules\framework\actions
python rl_generate_functions.py
```

This will use LLM to generate and refine reward functions for your task.

---

## 🛠️ Troubleshooting

### Python Not Found

**Error:** `'python' is not recognized as an internal or external command`

**Fix:**
1. Reinstall Python and check "Add Python to PATH"
2. Or manually add to PATH: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python310\`
3. Try using `py` instead of `python`: `py -m pip install ...`

### C++ Compilation Errors

**Error:** `CMake Error: Visual Studio not found`

**Fix:**
1. Install Visual Studio Build Tools with C++ workload
2. Or install full Visual Studio Community Edition
3. Restart PowerShell after installation

**Error:** `CMake not found`

**Fix:**
```powershell
winget install Kitware.CMake
```
Then restart PowerShell.

**Alternative:** Skip C++ compilation (slower performance)
- The environment will work without C++ acceleration, just slower
- You can proceed with training without compiling

### CUDA/GPU Issues

**Error:** `CUDA not available` but you have NVIDIA GPU

**Fix:**
1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Reinstall PyTorch with CUDA:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Missing Packages

**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Fix:**
```powershell
pip install xxx
```

Common missing packages:
```powershell
pip install pyyaml openai tenacity tqdm tensorboardX
```

### Path Issues

**Error:** `FileNotFoundError` or `cannot find file`

**Fix:**
- Use absolute paths in configuration files
- Use raw strings: `r'C:\path\to\file'`
- Or double backslashes: `'C:\\path\\to\\file'`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'cfg'` or `gym`

**Fix:**
1. Make sure you're in the correct directory
2. Set PYTHONPATH (see Step 8)
3. Or run from the `marl_llm` directory:
```powershell
cd C:\Users\YourUsername\Documents\MARL\marl_llm
python train/train_assembly.py
```

### Training Crashes

**Error:** Out of memory or training crashes

**Fix:**
1. Reduce batch size in `cfg/assembly_cfg.py`:
```python
parser.add_argument("--batch_size", default=1024, type=int)  # Reduced from 2048
```

2. Use CPU if GPU memory is insufficient:
```python
parser.add_argument("--device", default="cpu", type=str)
```

---

## 🎯 Quick Start Commands (Summary)

After installation, to start training:

```powershell
# Activate environment
.\marl_env\Scripts\activate  # OR: conda activate marl_llm

# Navigate to training directory
cd C:\Users\YourUsername\Documents\MARL\marl_llm\train

# Start training
python train_assembly.py

# In another terminal - monitor with TensorBoard
cd C:\Users\YourUsername\Documents\MARL\marl_llm\train\models\assembly
tensorboard --logdir=.
```

---

## 📁 Important File Locations

- **Training scripts:** `marl_llm/train/`
- **Configuration:** `marl_llm/cfg/assembly_cfg.py`
- **Saved models:** `marl_llm/train/models/assembly/YYYY-MM-DD-HH-MM-SS/`
- **Evaluation:** `marl_llm/eval/`
- **LLM config:** `marl_llm/llm/config/`
- **Target shapes:** `figures/` (you create this)

---

## 💡 Tips for Windows Users

1. **Use PowerShell or Windows Terminal** (better than CMD)
2. **Run as Administrator** if you encounter permission issues
3. **Disable antivirus temporarily** if it blocks Python packages
4. **Use absolute paths** in configuration files
5. **Check firewall** if downloading packages fails
6. **Use WSL2** (Windows Subsystem for Linux) as alternative if you encounter issues

---

## 📚 Additional Resources

- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **Visual Studio Build Tools:** https://visualstudio.microsoft.com/downloads/
- **CMake Tutorial:** https://cmake.org/cmake/help/latest/guide/tutorial/
- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html

---

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide first
2. Search existing issues: https://github.com/YourUsername/MARL/issues
3. Open a new issue with:
   - Your Windows version
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce

---

**Last Updated:** November 24, 2025  
**Tested on:** Windows 10/11, Python 3.10-3.13
