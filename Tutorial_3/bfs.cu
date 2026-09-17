#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void bfs( int V, const int *offsets, const int *edges, int *distance, int level) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V)
        return;

    // Process only vertices discovered at this level
    if (distance[u] != level)
        return;

    for (int i = offsets[u]; i < offsets[u + 1]; i++) {
        int v = edges[i];
        if (distance[v] == -1) {
            distance[v] = level + 1;
        }
    }
}

int main() {
    const int V = 8;
    const int E = 18;

int h_offsets[V + 1] =
{
    0,   // vertex 0: edges 0-1
    2,   // vertex 1: edges 2-4
    5,   // vertex 2: edges 5-6
    7,   // vertex 3: edges 7-8
    9,   // vertex 4: edges 9-11
    12,  // vertex 5: edges 12-14
    15,  // vertex 6: edges 15-16
    17,  // vertex 7: edge 17
    18   // end
};

int h_edges[E] =
{
    1, 2,          // 0
    0, 3, 4,       // 1
    0, 5,          // 2
    1, 4,          // 3
    1, 3, 5,       // 4
    2, 4, 6,       // 5
    5, 7,          // 6
    6              // 7
};

    int source = 0;

    int *d_offsets;
    int *d_edges;
    int *d_distance;

    cudaMalloc(&d_offsets, (V + 1) * sizeof(int));
    cudaMalloc(&d_edges, E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));

    cudaMemcpy(d_offsets,h_offsets,(V + 1) * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_edges,h_edges,E * sizeof(int),cudaMemcpyHostToDevice);

    // -1 = not visited
    cudaMemset(d_distance,-1,V * sizeof(int));

    // Source distance = 0
    int zero = 0;

    cudaMemcpy(&d_distance[source],&zero,sizeof(int),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (V + threadsPerBlock - 1) / threadsPerBlock;

    int max_level = 6
    for (int level = 0; level < max_level; level++) {

        bfs<<<blocks, threadsPerBlock>>>(
            V,
            d_offsets,
            d_edges,
            d_distance,
            level
        );

        cudaDeviceSynchronize();
    }

    vector<int> h_distance(V);

    cudaMemcpy(h_distance.data(),d_distance,V * sizeof(int),cudaMemcpyDeviceToHost);

    cout << "Shortest distances from vertex " << source << ":\n";

    for (int v = 0; v < V; v++) {
        cout << "Vertex " << v << ": " << h_distance[v] << '\n';
    }

    cudaFree(d_offsets);
    cudaFree(d_edges);
    cudaFree(d_distance);

    return 0;
}