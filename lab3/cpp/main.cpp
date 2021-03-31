#include <cstdlib>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "maxflow.h"

using namespace std;


// check if n_i, n_j is a neighbor of pixel i, j
bool neighbor_exist(
	int height,
	int width,
	int i,
	int j, 
	int n_i, 
	int n_j)
{

    if (n_i == i and n_j == j + 1 and 0 < n_j and n_j <= width -1)
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

template <typename T> 
class Matrix2d {
  private:
    std::valarray<T> ndarray;
    int height, width;
 public:
    Matrix2d(int height, int width) : ndarray(height*width),
                                      height(height),
                                      width(width) {}
    Matrix2d(T fill_value, int height, int width) : ndarray(fill_value, height*width),
                                      height(height),
                                      width(width) {}
    T& operator()(int row, int col) {return ndarray[col + row*width];}

    int get_shape(int dim) const
    {
        if (dim == 0)
        {
            return this->height;
        }
        else if (dim == 1)
        {
            return this->width;
        }
        else return -1;
    }

};


template <typename T> 
class Matrix3d {
  private:
    std::valarray<T> ndarray;
    int height, width, depth;
 public:
    Matrix3d(int height, int width, int depth) : ndarray(height*width*depth),
                                                 height(height),
                                                 width(width),
                                                 depth(depth) {}
    Matrix3d(T fill_value, int height, int width, int depth) : ndarray(fill_value,height*width*depth),
                                                 height(height),
                                                 width(width),
                                                 depth(depth) {}
    T& operator()(int row, int col, int k) {return ndarray[k + col*depth + row*width*depth];}

    int get_shape(int dim) const
    {
        if (dim == 0)
        {
            return this->height;
        }
        else if (dim == 1)
        {
            return this->width;
        }
        else if (dim == 2)
        {
            return this->depth;
        }
        else return -1;
    }

};



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
    while (current != source)
    {
        path.push_back(parent[current]);
        current = parent[current]->get_source();
    }
    return path;
}

// min-cut algorithm implementation
void restore_labelling(Residual_Graph &g, Matrix3d<int> &output, int source, int sink)
{   // run bfs with updated conditions
    int height = output.get_shape(0);
    int width = output.get_shape(1);
    int n_labels = output.get_shape(2);
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
                {   

                    // get pixel position from its number
                    int x = int((destination-1)/(width*n_labels));
                    int y = int((destination - 1)/(n_labels))%width;
                    int z = (destination -1)%n_labels;
                    // create label for pixel
                    output(x,y,z) = 1;

                }
            }
        }
        q.pop();
    }
}


int maxflow_mincut(Graph &g, Matrix3d<int> &output, int source, int sink)
{   // run maxflow algorithm
    int mx_flow = 0;
    Residual_Graph residual(g);

    auto curr_path = find_path(residual, source, sink);

    while (!curr_path.empty())
    {
        int b = std::numeric_limits<int>::max();
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

template <typename T> 
void calculate_q(
    Matrix3d<T> &q,
    const cv::Mat &image)
{
    // initialize unary penalties for 3 labels
    for(int i=0;i<q.get_shape(0);i++)
    {
        for(int j=0;j<q.get_shape(1);j++)
        {
        	int pixel_value = (int)image.at<uchar>(i,j);
        	q(i,j,0) = pixel_value;
        	q(i,j,1) = abs(128 - pixel_value);
        	q(i,j,2) = abs(255 - pixel_value);
        }
    }
}


void fill_graph(
    Graph &graph, 
    cv::Mat &image, 
    Matrix3d<int> &unary_penalties,
    int beta)
    {
        // fill graph with initial capacity values 
        int height = unary_penalties.get_shape(0);
        int width = unary_penalties.get_shape(1);
        int n_labels = unary_penalties.get_shape(2);
        int n_nodes = height * width * n_labels + 2;

        // add edges between labels in one pixel
        for(int i=0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {   
                for(int k=0;k<n_labels-1;k++)
                {
                    // map 3D->1D
                    int index = 1 + k + j*n_labels + i*width*n_labels;

                    graph.addEdge(index, index+1, 0, unary_penalties(i,j,k));
                    // reverse edge
                    graph.addEdge(index+1, index, 0, std::numeric_limits<int>::max());
                }
            }
        }


        // add parallel edges between neighbours (with capacity=beta)
        for(int i=0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {   
                for(int n_i=i; n_i<=i+1; n_i++)
                {
                    for(int n_j=j; n_j<=j+1; n_j++)
                    {
                        if (neighbor_exist(height,width,i,j,n_i,n_j))
                        {
                            for(int k=0;k<n_labels;k++)
                            {
                                int index = 1 + k + j*n_labels + i*width*n_labels;
                                int neighbor_index = 1 + k + n_j*n_labels + n_i*width*n_labels;

                                graph.addEdge(index, neighbor_index, 0, beta);
                                graph.addEdge(neighbor_index, index, 0, beta);
                            }
                        }
                    }
                }
            }
        }

        // add edges from source to each pixel (capacity=+inf)
        // edges from each pixel(last label) to sink (capacity=unary penalty for last label)
        for(int i=0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {   
                for(int k=0;k<n_labels;k++)
                {   
                    if (k==0)
                    {
                        int index = 1 + k + j*n_labels + i*width*n_labels;
                        // 0 - from source
                        graph.addEdge(0, index, 0, std::numeric_limits<int>::max());
                    }
                    if (k==n_labels-1)
                    {
                        int index = 1 + k + j*n_labels + i*width*n_labels;
                        // n_labels-1 - to sink
                        graph.addEdge(index, n_nodes-1, 0, unary_penalties(i,j,n_labels-1));
                    }
                }
            }
        }
    }

int main(int argc, char *argv[])
{

    // parse input parameters from command line
    string input_image_path = argv[1];
    int beta = atoi(argv[2]);
    string output_path = argv[3];

    // initialize all parameters for maxflow-mincut
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    int width = image.cols, height = image.rows;
    array<int, 3> labels{ 0, 1, 2};
    int n_labels = labels.size();

    // fill unary penalties
    Matrix3d<int> unary_penalties(0,height,width,n_labels);
    calculate_q(unary_penalties,image);


    // '+2' : source + sink
    int n_nodes = height * width * n_labels + 2;

    Graph graph(n_nodes);
    fill_graph(graph, image, unary_penalties, beta);

    // optimal labeling
    Matrix3d<int> labeling(height,width,n_labels);
    // run maxflow algoritm
    maxflow_mincut(graph, labeling,0, n_nodes-1);
    
    //create output image
    cv::Mat output = cv::Mat::zeros(height,width,CV_8UC1);
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {  
            // map labeling as sum of all labels in one object (i,j)
            int sum = 0;
            for(int k=0;k<n_labels;k++)
            {
                sum = sum + labeling(i,j,k);
            }
            output.at<uchar>(i,j) = (int)(255*((float)sum/(float)n_labels));
            
        }
    }
    // write result to file
     cv::imwrite(output_path, output);

    return 0;
}
