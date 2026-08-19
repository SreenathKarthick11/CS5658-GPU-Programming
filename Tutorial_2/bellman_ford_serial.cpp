#include <bits/stdc++.h>

using namespace std;

// Weighted CSR Graph
//
// offsets[v] to offsets[v + 1] gives the edges of vertex v.
//
// edges = [neighbour, weight, neighbour, weight, ...]

int main()
{
    const int V = 10;

    // Example graph:
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

    // Start index of each vertex's edges
    int offsets[V + 1] = {0,  2,  4,  6,  8, 10, 12, 13, 15, 17, 17};

    int edges[] = {1,4,2,1,3,1,4,7,1,2,5,8,4,3,6,6,6,2,7,5,4,4,7,2,8,1,9,10,8,3,9,6,9,2};

    const int E = 18;

    vector<int> distance(V, INT_MAX);

    int source = 0;
    distance[source] = 0;

    for (int i = 0; i < V - 1; i++){

        bool changed = false;

        for (int u = 0; u < V; u++){

            if (distance[u] == INT_MAX)
                continue;

            int start = offsets[u];
            int end   = offsets[u + 1];

            for (int i = start; i < end; i++){

                int v      = edges[2 * i];
                int weight = edges[2 * i + 1];

                if (distance[u] + weight < distance[v]){
                    distance[v] = distance[u] + weight;
                    changed = true;
                }
            }
        }

        if (!changed)
            break;
    }


    cout << "Shortest distances from vertex " << source << ":\n";

    for (int v = 0; v < V; v++) {
        cout << "Vertex " << v << ": ";
        if (distance[v] == INT_MAX)
            cout << "INT_MAX";
        else
            cout << distance[v];
        cout << '\n';
    }

    return 0;
}