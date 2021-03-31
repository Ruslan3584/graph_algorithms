#include "maxflow.h"
#include <bits/stdc++.h>
using namespace std;
Edge::Edge()
    {
        Edge(0, 0, 0, 0);
    }

Edge::Edge(int src, int des, int fl, int cap)
    {
        source = src;
        dest = des;
        flow = fl;
        capacity = cap;
    }

int Edge::get_dest() const
    {
    	return dest;
    }

int Edge::get_source() const
    {
    	return source;
    }

int Edge::get_flow() const
    {
    	return flow;
    }


int Edge::get_capacity() const
    {
    	return capacity;
    }

void Edge::set_flow(int f)
    {
    	this->flow = f;
    }
    
Residual_Edge::Residual_Edge() : Edge()
    {
        is_forward = false;
        counterpart = NULL;
    }

Residual_Edge::Residual_Edge(int src, int des, int fl, int cap, bool fwd) 
   : Edge(src, des, fl, cap)
    {
        is_forward = fwd;
        counterpart = NULL;
    }

void Residual_Edge::set_couterpart(Residual_Edge *n)
    {
        counterpart = n;
    }

Residual_Edge* Residual_Edge::get_counterpart() const
    {
    	return counterpart;
    }
    
Graph::Graph(int vertices)
    {
        numberOfNodes = vertices;
        adjacency_list = new vector<Edge *>[numberOfNodes];
    }

int Graph::get_number_nodes() const
    {
    	return numberOfNodes;
    }

Graph::~Graph()
    {
        for (int i = 0; i < numberOfNodes; i++)
        {
            for (auto it : adjacency_list[i])
            {
                delete it;
            }
        }
        delete[] adjacency_list;
    }

void Graph::addEdge(int v1, int v2, int flow, int capacity)
    {
        Edge *temp = new Edge(v1, v2, flow, capacity);
        adjacency_list[v1].push_back(temp);
    }

vector<Edge *> Graph::getNeighbours(int v)
    {
        return adjacency_list[v];
    }    
    
Residual_Graph::Residual_Graph(Graph &g)
    {
        adjacency_list = new vector<Residual_Edge *>[g.get_number_nodes()];
        n = g.get_number_nodes();
        for (int i = 0; i < n; i++)
        {
            for (auto it : g.getNeighbours(i))
            {
                Residual_Edge *N_fwd = new Residual_Edge(i, it->get_dest(), it->get_flow(), it->get_capacity(), true);
                Residual_Edge *N_rev = new Residual_Edge(it->get_dest(), i,  0, 0, false);
                N_fwd->set_couterpart(N_rev);
                N_rev->set_couterpart(N_fwd);
                adjacency_list[i].push_back(N_fwd);
                adjacency_list[it->get_dest()].push_back(N_rev);
            }
        }
    }

int Residual_Graph::get_number_nodes() const
    {
    	return n;
    }

vector<Residual_Edge *> Residual_Graph::getNeighbours(int v)
    {
        return adjacency_list[v];
    }

Residual_Graph::~Residual_Graph()
    {
        for (int i = 0; i < n; i++)
        {
            for (auto it : adjacency_list[i])
            {
                delete it;
            }
        }
        delete[] adjacency_list;
    }