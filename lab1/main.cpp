#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "maxflow.h"

using namespace std;


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
            if (!visited[destination] and out_edge->get_flow() != out_edge->get_capacity() and out_edge->get_capacity() > 0)
            {
                q.push(destination);
                parent[destination] = out_edge;
                visited[destination] = true;
            }
        }
        q.pop();
    }



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


int max_flow(Graph &g, int source, int sink)
{
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
        	int new_flow_bwd = edge->get_flow() - b;
            edge->set_flow(new_flow_fwd);
            edge->get_counterpart()->set_flow(new_flow_bwd);
        }
        curr_path = find_path(residual, source, sink);
    }
    return mx_flow;
}


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


void restore_labelling(Residual_Graph &g,cv::Mat &output, int source, int sink)
{
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
                {
                	int x = int((destination-1)/width);
                	int y = (destination - 1)%width;

                	output.at<uchar>(x,y) = 255;

                }

            }
        }
        q.pop();
    }



}




void test(Graph &g, cv::Mat &output, int source, int sink)
{
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
        	int new_flow_bwd = (edge->get_counterpart())
        	->get_flow() - b;
            edge->set_flow(new_flow_fwd);
            edge->get_counterpart()->set_flow(new_flow_bwd);
        }
        curr_path = find_path(residual, source, sink);
    }
    
    std::cout << mx_flow << std::endl;
    restore_labelling(residual,output,source,sink);



}



int main()
{
	

	cv::Mat image;
    // Mat grayImage;

	image = cv::imread("../test_images/a2.png", cv::IMREAD_GRAYSCALE);

    // Geting image parameters
    int width = image.cols, height = image.rows;




    int n_nodes = height * width + 2;

    Graph graph(n_nodes);

    // int counter = 1;
    for(int i=0;i<height;i++)
    {
    	for(int j=0;j<width;j++)
    	{   
    		int index = 1 + j + i * width;
    		graph.addEdge(0,index,0,(int)image.at<uchar>(i,j));
    		graph.addEdge(index,n_nodes-1,0,255 - (int)image.at<uchar>(i,j));



    		for(int n_i=i-1; n_i<=i+1; n_i++)
    		{
    			for(int n_j=j-1; n_j<=j+1; n_j++)
    			{


    				if (neighbor_exist(height,width,i,j,n_i,n_j))
    				{
    					int neighbor_index = 1 + n_j + n_i * height;
    					graph.addEdge(index,neighbor_index,0,50);
    				}
    			}
    		}


    		

    	}
     }



    cv::Mat output = cv::Mat::zeros(height,width,CV_8UC1);

    test(graph,output, 0, n_nodes-1);
	

    // cout << max_flow(graph, 0, n_nodes-1)<< std::endl;




    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", output);
    cv::waitKey(0);
    return 0;
}
