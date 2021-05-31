# Chen Lei 20M38073

## hpc_lecture
This is the final report for high performance computing lecture 2021


### How to use
Except cuda & mpi is on caculation node, all the other code are run on login node
In each file, result.txt shows the experiment results test on Tsubame

### Module required
```
module load cuda/11.2.146
module load openmpi
module load gcc/8.3.0
```


|  Name    | Total Time                           | GFlops    | Error
| -------- | ------------------------------------ | ----------|-------------- |
| mpi (Example)  | 0.224394 s                     | 0.149534  | 0.000016      |
| openmp & mpi  | 0.073221 s                      | 0.458262  | 0.000016      |
| cacheblocking & simd  | 0.001045 s               |**32.112608** | 0.000016  |
| cacheblocking & simd & openmp  | 0.000959 s      |**34.971330** |0.000016   |
|cuda & mpi                      |0.013852 s       |2.422394     |0.000016    |
