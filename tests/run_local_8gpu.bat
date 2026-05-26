@echo off
setlocal
cd /d "%~dp0.."

:: Local 8-GPU smoke test — simulates the full 3D topology on one machine.
:: Requires 8 CUDA-visible GPUs. For fewer GPUs, edit nproc_per_node and
:: adjust tp_size / pp_size / dp_size in config.py to match.
set USE_LIBUV=0
torchrun --standalone --nproc_per_node=8 train_3d.py %*