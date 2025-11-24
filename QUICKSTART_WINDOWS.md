# MARL-LLM Quick Start Guide (Windows)

## ⚡ Installation (5 Steps)

```powershell
# 1. Clone repository
git clone https://github.com/YourUsername/MARL.git
cd MARL

# 2. Create and activate virtual environment
python -m venv marl_env
.\marl_env\Scripts\activate

# 3. Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install custom gym environment
cd cus_gym
pip install -e .

# 5. Install MARL-LLM dependencies
cd ..\marl_llm
pip install -r requirements.txt
```

## 🎮 Running Training (2 Steps)

```powershell
# 1. Update image folder path in cfg\assembly_cfg.py
# Change line ~163: image_folder = r'C:\path\to\your\figures'

# 2. Run training
cd train
python train_assembly.py
```

## 📊 View Results

```powershell
# Start TensorBoard (in new terminal)
cd marl_llm\train\models\assembly
tensorboard --logdir=.

# Open browser: http://localhost:6006
```

## 🧪 Evaluate Model

```powershell
# 1. Note your experiment timestamp folder name
# Example: 2025-11-24-14-30-25

# 2. Edit eval\eval_assembly.py
# Change line ~14: curr_run = '2025-11-24-14-30-25'

# 3. Run evaluation
cd marl_llm\eval
python eval_assembly.py
```

## 🔧 Common Issues

| Issue | Solution |
|-------|----------|
| `python not found` | Add Python to PATH or use `py` |
| `CUDA not available` | Install CUDA Toolkit + reinstall PyTorch |
| `ModuleNotFoundError` | Run `pip install <module>` |
| `CMake error` | Install Visual Studio Build Tools |
| `Out of memory` | Reduce batch_size in cfg file |

## 📂 Key Files

- **Config:** `marl_llm\cfg\assembly_cfg.py`
- **Training:** `marl_llm\train\train_assembly.py`
- **Evaluation:** `marl_llm\eval\eval_assembly.py`
- **Models:** `marl_llm\train\models\assembly\`

## 💡 Pro Tips

✅ Use absolute paths: `r'C:\Users\...'`  
✅ Run PowerShell as Administrator  
✅ Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`  
✅ Monitor training: Use TensorBoard in parallel  
✅ Save often: Models auto-save every 4 intervals  

---

**Full Guide:** See WINDOWS_SETUP.md
