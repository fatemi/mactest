# MacTest
A barebone codebase for GPU test


# Usage

GPU on M1 or M2 chips:

```
python mac_test.py -d mps
```

CPU test:

```
python mac_test.py -d cpu
```

On M2 base model [CPU: 10 cores (6 performance and 4 efficiency) + GPU: 16 cores] + Metal3:

mps: 

1st run = `~3.20 sec`

2nd+ runs = `~1.80 sec`

cpu (all runs) = `~25.70 sec`
