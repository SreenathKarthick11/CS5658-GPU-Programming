#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// Weighted CSR Graph
//
// offsets[v] to offsets[v + 1] gives the edges of vertex v.
//
// edges = [neighbour, weight, neighbour, weight, ...]
//
// Example:
// 0 -> 1 (4), 0 -> 2 (1)
// 1 -> 3 (1), 1 -> 4 (7)
// 2 -> 1 (2), 2 -> 5 (8)
// 3 -> 4 (3), 3 -> 6 (6)
// 4 -> 6 (2), 4 -> 7 (5)
// 5 -> 4 (4), 5 -> 7 (2)
// 6 -> 8 (1)
// 7 -> 8 (3), 7 -> 9 (6)
// 8 -> 9 (2)
// 9 -> nothing


__global__ void bellmanFord(int V,const int *offsets,const int *edges,int *distance,bool *changed){
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V)
        return;

    if (distance[u] == INT_MAX)
        return;

    int start = offsets[u];
    int end   = offsets[u + 1];

    for (int i = start; i < end; i++){

        int v      = edges[2 * i];
        int weight = edges[2 * i + 1];

        int newDistance = distance[u] + weight;

        if (newDistance < distance[v]){
            atomicMin(&distance[v], newDistance);
            *changed = true;
        }
    }
}


int main(){

    const int V = 10;

    int h_offsets[V + 1] = {0,  2,  4,  6,  8, 10, 12, 13, 15, 17, 17};

    int h_edges[] = {1,4,2,1,3,1,4,7,1,2,5,8,4,3,6,6,6,2,7,5,4,4,7,2,8,1,8,3,9,6,9,2};

    const int E = 17;


    vector<int> h_distance(V, INT_MAX);

    int source = 0;
    h_distance[source] = 0;

    int *d_offsets;
    int *d_edges;
    int *d_distance;
    bool *d_changed;

    cudaMalloc(&d_offsets, (V + 1) * sizeof(int));
    cudaMalloc(&d_edges, 2 * E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));


    cudaMemcpy(d_offsets,h_offsets,(V + 1) * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_edges,h_edges,2 * E * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_distance,h_distance.data(),V * sizeof(int),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (V + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < V - 1; i++)
    {
        bool h_changed = false;

        cudaMemcpy(d_changed,&h_changed,sizeof(bool),cudaMemcpyHostToDevice);

        bellmanFord<<<blocks, threadsPerBlock>>>(V,d_offsets,d_edges,d_distance,d_changed);

        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed,d_changed,sizeof(bool),cudaMemcpyDeviceToHost);

        if (!h_changed)
            break;
    }

    cudaMemcpy(h_distance.data(),d_distance,V * sizeof(int),cudaMemcpyDeviceToHost);

    cout << "Shortest distances from vertex "
         << source << ":\n";

    for (int v = 0; v < V; v++){
        cout << "Vertex " << v << ": ";

        if (h_distance[v] == INT_MAX)
            cout << "INT_MAX";
        else
            cout << h_distance[v];

        cout << '\n';
    }

    cudaFree(d_offsets);
    cudaFree(d_edges);
    cudaFree(d_distance);
    cudaFree(d_changed);

    return 0;
}

// OUTPUT:
// Shortest distances from vertex 0:
// Vertex 0: 0
// Vertex 1: 3
// Vertex 2: 1
// Vertex 3: 4
// Vertex 4: 7
// Vertex 5: 9
// Vertex 6: 9
// Vertex 7: 11
// Vertex 8: 10
// Vertex 9: 12