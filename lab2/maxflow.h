#include <bits/stdc++.h>
using namespace std;
class Edge
{ 
  // class stores edges for graph
  private:
    // source --> dest with flow and capacity
    int dest;
    int source;
    int flow;
    int capacity;
 public:
    // create edge with capacity and flow = 0
    Edge();
    int get_dest();
    int get_source();
    int get_flow();
    int get_capacity();
    void set_flow(int f);
    Edge(int src, int des, int fl, int cap);
};

class Residual_Edge : public Edge
{
  // class of residual edges of a graph
  private:
    // direction
    bool is_forward;
    // edge with reversed direction
    Residual_Edge *counterpart;
  public:
    Residual_Edge();
    Residual_Edge(int src, int des, int fl, int cap, bool fwd);
    // create reversed edge
    void set_couterpart(Residual_Edge *n);
    Residual_Edge* get_counterpart();
};

class Graph
{
  // graph as a adjacency list
  private:
    vector<Edge *> *adjacency_list;
    int numberOfNodes;
  public:
    Graph(int vertices);
    ~Graph();
    int get_number_nodes();
    void addEdge(int v1, int v2, int flow, int capacity);
    vector<Edge *> getNeighbours(int v);
};

class Residual_Graph
{
  // stores residual capacities for graph
  private:
  	int n;
    vector<Residual_Edge *> *adjacency_list;
  public:
    Residual_Graph(Graph &g);
    int get_number_nodes();
    ~Residual_Graph();
    vector<Residual_Edge *> getNeighbours(int v);
};