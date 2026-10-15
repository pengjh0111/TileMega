"""P-1 driver: load the probe .so into a torch process and check results."""
import ctypes, sys, torch
lib = ctypes.CDLL(sys.argv[1])
n = 1000
x = torch.arange(n, dtype=torch.float32, device="cuda")
ok = True
for mode in (0, 1):
    y = torch.zeros_like(x)
    info = (ctypes.c_int * 8)()
    lib.probe_run(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), n, mode,
                  ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), info)
    torch.cuda.synchronize()
    if mode == 0:
        ref = x * 2
    else:
        ref = torch.empty_like(x)
        for b in range(0, n, 256):
            blk = torch.zeros(256, device="cuda"); m = min(256, n - b); blk[:m] = x[b:b + m]
            ref[b:b + m] = blk.flip(0)[:m] + 1
    good = torch.equal(y, ref); ok &= good
    print(f"mode={mode} setattr={info[0]} occ_err={info[1]} occ_blocks={info[2]} attr_err={info[3]} "
          f"regs={info[4]} local={info[5]} maxthr={info[6]} launch={info[7]} correct={good}")
print("P1", "PASS" if ok else "FAIL")
