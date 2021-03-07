#include <bits/stdc++.h>
using namespace std;


class Edge
{
  private:
    int dest;
    int source;
    int flow;
    int capacity;
 public:

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
  
   private:

    bool is_forward;
    Residual_Edge *counterpart;

  public:

    Residual_Edge();
    Residual_Edge(int src, int des, int fl, int cap, bool fwd);
    void set_couterpart(Residual_Edge *n);
    Residual_Edge* get_counterpart();
};



class Graph
{
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
  private:
  	int n;
    vector<Residual_Edge *> *adjacency_list;
  public:
    Residual_Graph(Graph &g);
    int get_number_nodes();
    ~Residual_Graph();
    vector<Residual_Edge *> getNeighbours(int v);
};









