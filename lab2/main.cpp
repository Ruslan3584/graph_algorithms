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

    int get_shape(int dim)
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

    int get_shape(int dim)
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
    while (!current == source)
    {
        path.push_back(parent[current]);
        current = parent[current]->get_source();
    }
    return path;
}

// min-cut algorithm implementation
void restore_labelling(Residual_Graph &g,Matrix2d<int> &output, int source, int sink)
{   // run bfs with updated conditions
    int height = output.get_shape(0);
    int width = output.get_shape(1);

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
                    output(x,y) = 1;
                }
            }
        }
        q.pop();
    }
}


int maxflow_mincut(Graph &g, Matrix2d<int> &output, int source, int sink)
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



// g(x,y) = epsilon*1(x!=y)
template <typename T> 
void calculate_g(
    Matrix2d<T> &g,
    int epsilon)
{
	int n_labels = g.get_shape(0);

    for(int i=0;i<n_labels;i++)
    {
        for(int j=0;j<n_labels;j++)
        {
            g(i,j) = epsilon * int(i!=j);
        }
    }
}



template <typename T> 
void calculate_q(
    Matrix3d<T> &q,
    cv::Mat &image)
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



template <typename T> 
void reparametrization(
	Matrix2d<int> &labeling,
	int alpha,
	Matrix3d<T> &unary_penalties,
	Matrix2d<T> &binary_penalties,
	Matrix3d<T> &reparametrized_Q
	)
{
    // reparametrization process
	int height = labeling.get_shape(0);
	int width = labeling.get_shape(1);
	{   
        // for every pixel pair
		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				int k = labeling(i,j);
                // for every neighbour
				for(int n_i=i; n_i<=i+1; n_i++)
	            {
	                for(int n_j=j; n_j<=j+1; n_j++)
	                {
	                    if (neighbor_exist(height,width,i,j,n_i,n_j))
	                    {
	                        int kn = labeling(n_i,n_j);
	                        
	                        reparametrized_Q(i,j,0) = unary_penalties(i,j,k);
                			reparametrized_Q(i,j,1) = unary_penalties(i,j,alpha) + binary_penalties(alpha, alpha) 
                																 - binary_penalties(k, alpha);
                			reparametrized_Q(n_i,n_j,0) = unary_penalties(n_i,n_j,k) + binary_penalties(k, kn);
                			reparametrized_Q(n_i,n_j,1) = unary_penalties(n_i,n_j,alpha) + binary_penalties(k, alpha);
	                    }
	                }
	            }

			}
		}
	}
}



// labeling(t) = argmin_k(q_t(k))
void initialize_labeling(
	Matrix3d<int> &unary_penalties,
	Matrix2d<int> &labeling)
{
	int height = unary_penalties.get_shape(0);
	int width = unary_penalties.get_shape(1);
	int n_labels = unary_penalties.get_shape(2);
	for(int i=0;i<height;i++)
    {
    	for(int j=0;j<width;j++)
    	{
    		int min_value = std::numeric_limits<int>::max();
    		int argmin;
    		for(int k=0;k<n_labels;k++)
    		{
    			if(unary_penalties(i,j,k) <= min_value)
    			{
    				min_value = unary_penalties(i,j,k);
    				argmin = k;
    			}
    			
    		}
    		labeling(i,j) = argmin;
    	}
    }
}




void update_labeling(
	Matrix3d<int> &reparametrized_Q,
	Matrix2d<int> &binary_penalties,
	Matrix2d<int> &labeling,
	int alpha,
	Matrix2d<int> &segments
	)
{   
    // one iteration of maxflow-mincut algorithm for new reparametrized penalties
	int height = labeling.get_shape(0);
	int width = labeling.get_shape(1);
	int n_nodes = height * width + 2;
	Graph graph(n_nodes);

    
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {   

        	int k = labeling(i,j);
            int index = 1 + j + i * width;
            // add weights for 'source' and 'sink' nodes
            // 0 node is a source, n_nodes-1 node is a sink
            graph.addEdge(0, index, 0, reparametrized_Q(i,j,0));
            graph.addEdge(index, n_nodes-1, 0, reparametrized_Q(i,j,1));
            // add weights beetwen pixels
            // pixels nodes are numbers from 1 to n_nodes-2
            for(int n_i=i; n_i<=i+1; n_i++)
            {
                for(int n_j=j; n_j<=j+1; n_j++)
                {
                    if (neighbor_exist(height,width,i,j,n_i,n_j))
                    {
                        int neighbor_index = 1 + n_j + n_i * width;

                        int kn = labeling(n_i,n_j);
	                    int neighbor_penalty = binary_penalties(alpha, kn) + binary_penalties(k, alpha)
	                    			          - binary_penalties(k, kn) - binary_penalties(alpha, alpha);

                        graph.addEdge(index, neighbor_index, 0, neighbor_penalty);
                    }
                }
            }
        }
     }
     // calculate maxflow value
     int maxflow = maxflow_mincut(graph, segments, 0, n_nodes-1);
}


Matrix2d<int> alpha_expansion_step(
    Matrix2d<int> &labeling,
    array<int, 3> &labels,
    Matrix3d<int> &unary_penalties,
    Matrix2d<int> &binary_penalties
)
{   
    // iteration of alpha-expansion algorithm
    int height = labeling.get_shape(0);
	int width = labeling.get_shape(1);
    int n_labels = labels.size();
    // shuffle alpha labels
    std::shuffle(labels.begin(), labels.end(), default_random_engine(0));
    // create labelling
    Matrix2d<int> k_prev = labeling;
    // temp reparametrization matrix
    Matrix3d<int> reparametrized_Q(0,height,width,2);
    for(auto&& alpha : labels)
    {   
        // reparametrize penalties
	    reparametrization(k_prev,alpha,unary_penalties,binary_penalties,reparametrized_Q);
        Matrix2d<int> k_next(0,height,width);
        update_labeling(reparametrized_Q,binary_penalties,k_prev,alpha,k_next);
        // remap labelling with alpha
        for(int i=0;i<height;i++)
        {
            for (int j = 0; j < width; j++)
            {
                if(k_next(i,j)==1)
                {
                    k_prev(i,j) = alpha;
                }                
            }
        }
    }
    return k_prev;
}


int main(int argc, char *argv[])
{

    // parse input parameters from command line
    string input_image_path = argv[1];
    int epsilon = atoi(argv[2]);
    int n_iter = atoi(argv[3]);
    string output_path = argv[4];

    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    int width = image.cols, height = image.rows;
    array<int, 3> labels{ 0, 1, 2};
    int n_labels = labels.size();

    Matrix3d<int> unary_penalties(0,height,width,n_labels);
    calculate_q(unary_penalties,image);
    Matrix2d<int> binary_penalties(0,n_labels,n_labels);
    calculate_g(binary_penalties,epsilon);
    // k_init
    Matrix2d<int> labeling(0,height,width);
    initialize_labeling(unary_penalties,labeling);

    // run alpha-expansion n_iter times
    for(int iter=0;iter<n_iter;iter++)
    {
        labeling = alpha_expansion_step(labeling,labels,unary_penalties,binary_penalties);
    }
    
    // create output image
    cv::Mat output = cv::Mat::zeros(height,width,CV_8UC1);
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {  
        	if(labeling(i,j)==0)
        	{
        		output.at<uchar>(i,j) = 0;
        	}
            else if(labeling(i,j)==1)
            {
                output.at<uchar>(i,j) = 128;
            }
            else if(labeling(i,j)==2)
            {
                output.at<uchar>(i,j) = 255;
            }
        	
        }
    }

    cv::imwrite(output_path, output);

    return 0;
}