Device: NVIDIA GeForce RTX 4060 Laptop GPU
Precision: torch.bfloat16
Batch size: 1
Warmup: 5
Repetitions: 20

Seq Len     Mean (ms)      Median (ms)    P95 (ms)       
---------------------------------------------------------
16          24.050         24.049         24.977         
32          24.296         24.324         24.841         
64          25.940         25.816         26.651         
128         34.344         34.343         34.892         


Device: cuda:0
Precision: torch.bfloat16
Prompt length: 32
Batch size: 1
Warmup: 2

New Tokens    TTFT (ms)     TPOT (ms)     Total (s)     Throughput (tok/s)    
------------------------------------------------------------------------------
16            25.303        25.335        0.405         39.47                 
32            24.795        25.490        0.815         39.26                 
64            25.310        29.592        1.890         33.86                 
