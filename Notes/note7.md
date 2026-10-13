# Notes of GPU Programming

```text
Date : 13 Oct 2026
```

## How CUDA code is compiled ?

The below is a flowchart that shows how an cuda program in compiled.

```mermaid
flowchart TD

subgraph F["Fat Binary"]
    direction TD
    F1["PTX <br> Compute 8.0"]
    F2["PTX <br> Compute 9.0"]
    F3["CuBin SM_8.0"]
    F4["CuBin SM_8.5"]
    F5["CuBin SM_9.0"]
end

H[hello.cu] --> N[NVCC]
N --> Ho[Host Compiler]
N --> G[GPU Compiler]
Ho --> HoCu["Host CPU <br> Binary Code"]
HoCu --> F

G --> P1["PTX <br> Compute 8.0"]
G --> P2["PTX <br> Compute 9.0"]
P1 --> Pa1[PTX assembly]
P2 --> Pa2[PTX assembly]
Pa1 --> Cu1[CuBin SM_8.0]
Pa1 --> Cu2[CuBin SM_8.5]
Pa2 --> Cu3[CuBin SM_9.0]
```
> [!NOTE]
> PTX is virtual GPU code

---

## CUDA OBJECT DUMP

`cudaobjdump` is an NVIDIA CUDA Toolkit command-line utility used to inspect, disassemble, and extract low-level device code from compiled CUDA binaries (.o, .a, .fatbin, and executables).

| Flag | Description / Output |
| --- | --- |
| **`-sass`** | Native GPU machine assembly (SASS), register usage, & instruction scheduling |
| **`-ptx`** | Architecture-independent intermediate assembly (PTX) |
| **`-source`** | High-level CUDA C/C++ source interleaved with SASS (compiled with `-lineinfo`) |
| **`-elf`** | GPU ELF metadata: symbol tables, sections, shared memory, & register allocation |
| **`-fatbin`** | Fatbinary container details and target GPU architectures (e.g., `sm_80`, `sm_90`) |

```bash
cudaobjdump -sass my_app                      # Dump native GPU assembly
cudaobjdump -source my_app                    # Interleave C++ source with SASS
cudaobjdump -gpu sm_80 -sass my_app           # Target specific GPU architecture
```

> [!NOTE]
> With `nvcc` , you can compile your code to compute capability to 8.0 on a machine that has GPU compute capability 7.0 , but you wont be able to run

---

## Some Questions

> [!QUESTION] How do we display all architectures supported by NVCC?
>
> **Virtual architectures (`compute_XX`):**
> ```bash
> nvcc --list-gpu-arch
> ```
>
> **Real GPU device codes (`sm_XX`):**
> ```bash
> nvcc --list-gpu-code
> ```


> [!QUESTION] `nvcc -ptx code.cu` what is the output of this command
> It compiles `code.cu` into human-readable **PTX assembly** (architecture-independent intermediate representation) and generates a file named **`code.ptx`** in the current directory. It does not produce an executable or object file.


---
