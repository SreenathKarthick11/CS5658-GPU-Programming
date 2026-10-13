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
    F3["CuBin <br> sm_8.0"]
    F4["CuBin <br> sm_8.6"]
    F5["CuBin <br> sm_9.0"]
end

H(hello.cu) --> N[NVCC]
N --> Ho[Host Compiler]
N --> G[GPU Compiler]
Ho --> HoCu["Host CPU <br> Binary Code"]
HoCu --> F

G --> P1["PTX <br> Compute 8.0"]
G --> P2["PTX <br> Compute 9.0"]
P1 --> Pa1[PTX assembly]
P2 --> Pa2[PTX assembly]
Pa1 --> Cu1[CuBin <br> sm_8.0]
Pa1 --> Cu2[CuBin <br> sm_8.6]
Pa2 --> Cu3[CuBin <br> sm_9.0]
```
> [!NOTE]
> PTX is virtual GPU code

The below is refined diagram by **Gemini**.

```mermaid
flowchart TD
    H["hello.cu"] --> N["NVCC Driver"]

    %% Split
    N --> Ho["Host Compiler (GCC / MSVC)"]
    N --> G["GPU Frontend Compiler (cicc)"]

    %% Host Path
    Ho --> HoObj["Host CPU Object Code"]

    %% GPU Path - Intermediate PTX
    G --> P1["PTX (compute_80)"]
    G --> P2["PTX (compute_90)"]

    %% ptxas Assembler
    P1 -->|ptxas| Cu1["CuBin (sm_80)"]
    P1 -->|ptxas| Cu2["CuBin (sm_86)"]
    P2 -->|ptxas| Cu3["CuBin (sm_90)"]

    %% Fatbinary Bundle
    subgraph F["Fat Binary (.fatbin)"]
        P1
        P2
        Cu1
        Cu2
        Cu3
    end

    %% Final Embed & Link
    F --> Link["Host Linker"]
    HoObj --> Link
    Link --> Exec["Final Executable"]
```
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
> It compiles `code.cu` into  **PTX intermedate** (architecture-independent intermediate representation) and generates a file named **`code.ptx`** in the current directory. It does not produce an executable or object file.

>[!NOTE]
 `reduce_add_sync` can only run in GPU with compute capability 8.0 or higher.


---
