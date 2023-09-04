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

On `M2 Pro` base model [CPU: 10 cores (6 performance and 4 efficiency) + GPU: 16 cores] + Metal3: 

cpu (all runs) = `~25.70 sec`

mps 1st run = `~3.20 sec`

mps 2nd+ runs = `~1.80 sec`


On `M1 Pro` base model [CPU: 8 cores (6 performance and 2 efficiency) + GPU: 14 cores] + Metal3: (thanks to Alireza)

cpu = `25.7--26.5 sec`

mps 1st run = `~4.51 sec`

mps 2nd+ runs = `~2.32 sec`
