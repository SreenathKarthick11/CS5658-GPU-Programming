# Tutorial 2



1) Implement SSSP on GPU with graph stored in CSR format.Graph can be read from hard disk to CPU RAM and store it in  CSR format.

**Solution:** [bellman_ford.cu](bellman_ford.cu)

2) Try to implement  a serial algorithm on CPU, check its correctness.

**Solution:** [bellman_ford_serial.cpp](bellman_ford_serial.cpp)


>**Note:** The following graph is used for verify the implementation of SSSP algorithm.


```mermaid
graph LR
    0((0)) -->|4| 1((1))
    0 -->|1| 2((2))

    1 -->|1| 3((3))
    1 -->|7| 4((4))

    2 -->|2| 1
    2 -->|8| 5((5))

    3 -->|3| 4
    3 -->|6| 6((6))

    4 -->|2| 6
    4 -->|5| 7((7))

    5 -->|4| 4
    5 -->|2| 7

    6 -->|1| 8((8))

    7 -->|3| 8
    7 -->|6| 9((9))

    8 -->|2| 9
```
