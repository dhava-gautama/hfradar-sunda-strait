import os, sys, time, torch
sys.path.insert(0, os.path.expanduser("~/radarMaritim/scripts"))
from phase3_seeds import ConvLSTMED, BiEF, CNNGRU_MS
torch.set_num_threads(4)  # representative of a modest operational CPU
ny, nx, H, T = 21, 21, 6, 12
dev = torch.device("cpu")  # operational target is CPU
models = {
    "CNN-GRU-MS-Matched (direct,0.48M)": CNNGRU_MS(ny, nx, H, 40),
    "CNN-GRU-MS full (direct,2.85M)":    CNNGRU_MS(ny, nx, H, 256),
    "BiEF (autoregressive,0.47M)":       BiEF(ny, nx, H, 64),
    "ConvLSTM-ED (autoregressive,0.30M)":ConvLSTMED(ny, nx, H, 64),
}
x = torch.randn(1, T, 2, ny, nx)  # one operational forecast (batch=1)
print(f"{'model':38} {'ms/forecast (CPU, batch=1)':>26}")
for name, m in models.items():
    m.eval()
    with torch.no_grad():
        for _ in range(3): m(x)            # warmup
        t0 = time.time()
        for _ in range(50): m(x)
        ms = (time.time()-t0)/50*1000
    print(f"{name:38} {ms:>22.1f} ms")
