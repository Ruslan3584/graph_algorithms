import numpy as np
from numba import njit
from skimage.io import imread, imsave
import maxflow
import argparse

@njit
def get_right_down(height,width,i,j):

    # i,j - position of pixel
    # [Right, Down] - order of possible neighbours
    # array of neighbour indices
    nbs = [] 
    # Right
    if 0<j+1<=width-1 and 0<=i<=height-1:
        nbs.append([i,j+1])
    # Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        nbs.append([i+1,j])
    return nbs


def fill_graph(Q,beta):

    height, width, n_labels = Q.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((height, width,n_labels))
    
    # add edges inside object
    for i in range(height):
        for j in range(width):
            for k in range(n_labels-1):
                g.add_edge( nodeids[i, j,k], nodeids[i,j,k+1], Q[i,j,k], np.inf)

    # add parellel edges between neighbours
    for i in range(height):
        for j in range(width):
            neighbours = get_right_down(height,width,i,j)
            for [ni,nj] in neighbours:
                for k in range(n_labels):
                    g.add_edge( nodeids[i, j,k], nodeids[ni,nj,k], beta, beta)
    
    # add terminal edges
    g.add_grid_tedges(nodeids[...,0],np.inf,0)
    g.add_grid_tedges(nodeids[...,n_labels-1], 0, Q[...,n_labels-1])
    
    return g, nodeids


def ishikawa_maxflow(image,beta,n_labels):

    height, width = image.shape
    labels = np.linspace(0,255,n_labels).astype(int)

    # define unary penalties
    Q = np.abs(image[..., None] - labels)
    g, nodeids = fill_graph(Q,beta)
    g.maxflow()
    segments = g.get_grid_segments(nodeids)
    labelling = np.int_(np.logical_not(segments))
    # remap labelling into 0-255
    output_img = labels[np.sum(labelling,axis=2)-1]
    return output_img


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str, help="path to input image")
    parser.add_argument("beta", type=int, help="binary penalty")
    parser.add_argument("n_labels", type=int, help="number of labels (segments)")
    parser.add_argument("output_image", type=str, help="path to output image")
    args = parser.parse_args()

    img = imread(args.input_image).astype(int)

    out = ishikawa_maxflow(img,args.beta,args.n_labels).astype(np.uint8)

    
    imsave(args.output_image, out)
    


if __name__ == "__main__":
    main()