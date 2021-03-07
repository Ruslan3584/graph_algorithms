#include <cstdlib>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "maxflow.h"

using namespace std;

// breadth-first search for edmonds karp algorithm
vector<Residual_Edge *> find_path(Residual_Graph &g, int source, int sink)
{   

    int numNodes = g.get_number_nodes();
    vector<Residual_Edge *> path;
    queue<int> q;
    bool visited[numNodes];
    Residual_Edge *parent[numNodes];

    for (int i = 0; i < numNodes; i++)
    {
        visited[i] = false;
    }

    visited[source] = true;
    q.push(source);

    while ((!q.empty()) and !visited[sink])
    {
        for (auto out_edge : g.getNeighbours(q.front()))
        {
            int destination = out_edge->get_dest();
            if (!visited[destination] and out_edge->get_flow() != out_edge->get_capacity()
                    and out_edge->get_capacity() > 0)
            {
                // update queue
                q.push(destination);
                // add to path
                parent[destination] = out_edge;
                // add to visited
                visited[destination] = true;
            }
        }
        q.pop();
    }
    // restore path from source to sink
    if (!visited[sink])
        return path;
    int current = sink;
    while (!current == source)
    {
        path.push_back(parent[current]);
        current = parent[current]->get_source();
    }
    return path;
}

// check if n_i, n_j is a neighbor of pixel i, j
bool neighbor_exist(int height, int width, int i, int j, int n_i, int n_j)
{
    if (n_i == i and n_j == j - 1 and 0 <= n_j and n_j < width -1)
    {
        return true;
    }
    else if (n_i == i and n_j == j + 1 and 0 < n_j and n_j <= width -1)
    {
        return true;
    }
    else if (n_i == i-1 and n_j == j and 0 <= n_i and n_i < height - 1)
    {
        return true;
    }
    else if (n_i == i+1 and n_j == j and 0 < n_i and n_i <= height - 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// min-cut algorithm implementation
void restore_labelling(Residual_Graph &g,cv::Mat &output, int source, int sink)
{   // run bfs with updated conditions
    int width = output.cols, height = output.rows;

    queue<int> q;
    bool visited[g.get_number_nodes()];
    for (int i = 0; i < g.get_number_nodes(); i++)
    {
        visited[i] = false;
    }
    visited[source] = true;
    q.push(source);



    while ((!q.empty()) and !visited[sink])
    {
        for (auto out_edge : g.getNeighbours(q.front()))
        {   
            int destination = out_edge->get_dest();
            if (!visited[destination] and out_edge->get_capacity() > out_edge->get_flow() and out_edge->get_capacity() > 0)
            {
                q.push(destination);
              
                visited[destination] = true;
                if (destination != source and destination != sink)
                {   // get pixel position from its number
                    int x = int((destination-1)/width);
                    int y = (destination - 1)%width;
                    // create label for pixel
                    output.at<uchar>(x,y) = 255;
                }
            }
        }
        q.pop();
    }
}


int maxflow_mincut(Graph &g, cv::Mat &output, int source, int sink)
{   // run maxflow algorithm
    int mx_flow = 0;
    Residual_Graph residual(g);

    auto curr_path = find_path(residual, source, sink);

    while (!curr_path.empty())
    {
        int b = INT32_MAX;
        for (auto edge : curr_path)
        {
            b = min(b, edge->get_capacity() - edge->get_flow());
        }
        mx_flow = mx_flow + b;

        for (auto edge : curr_path)
        {   
            int new_flow_fwd = edge->get_flow() + b;
            int new_flow_bwd = (edge->get_counterpart())->get_flow() - b;
            edge->set_flow(new_flow_fwd);
            edge->get_counterpart()->set_flow(new_flow_bwd);
        }
        curr_path = find_path(residual, source, sink);
    }
    // run min-cut
    restore_labelling(residual,output,source,sink);

    return mx_flow;
}

void fill_graph(Graph &graph, cv::Mat &image, int neighbor_penalty)
{   
    // fill graph with initial capacity values 
    int width = image.cols, height = image.rows;
    int n_nodes = height * width + 2;

    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {   
            int index = 1 + j + i * width;
            // add weights for 'source' and 'sink' nodes
            // 0 node is a source, n_nodes-1 node is a sink
            graph.addEdge(0, index, 0, (int)image.at<uchar>(i,j));
            graph.addEdge(index, n_nodes-1, 0, 255 - (int)image.at<uchar>(i,j));
            // add weights beetwen pixels
            // pixels nodes are numbers from 1 to n_nodes-2
            for(int n_i=i-1; n_i<=i+1; n_i++)
            {
                for(int n_j=j-1; n_j<=j+1; n_j++)
                {
                    if (neighbor_exist(height,width,i,j,n_i,n_j))
                    {
                        int neighbor_index = 1 + n_j + n_i * height;
                        graph.addEdge(index, neighbor_index, 0, neighbor_penalty);
                    }
                }
            }
        }
     }
}

int main(int argc, char *argv[])
{
    
    // parse input parameters from command line
    string input_image_path = argv[1];
    int neighbor_penalty = atoi(argv[2]);
    string output_path = argv[3];

    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
    // Getting image parameters
    int width = image.cols, height = image.rows;
    int n_nodes = height * width + 2;
    // create a graph with nodes
    Graph graph(n_nodes);
    fill_graph(graph, image, neighbor_penalty);

    cv::Mat output = cv::Mat::zeros(height,width,CV_8UC1);
    int max_flow = maxflow_mincut(graph, output,0, n_nodes-1);

    cout << "MaxFlow is: " << max_flow << endl;
    cv::imwrite(output_path, output);

    return 0;
}