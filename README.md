# Graph Algorithms in Computer Vision
Labs for University Course     

## Lab 1 - MaxFlow-MinCut implementation
### Image Denoising
#### Examples
```bash
cd lab1/
mkdir build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -mtune=native" .. && cmake --build .
```
pass parameters:
`path_to_input_image binary_penalty output_image`

run: 
```bash
./maxflow ../test_images/input.png 50 ../test_images/result.png
```
## Lab 2 - alpha-Expansion implementation
### Image Segmentation
#### Examples
```bash
cd lab2/
mkdir build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -mtune=native" .. && cmake --build .
```
pass parameters:
`path_to_input_image epsilon n_iter output_image`

run: 
```bash
./alpha_expansion ../input.png 15 3 ../out.png
```
## Lab 3 - MaxFlow implementation (Hiroshi Ishikawa)
### Image Segmentation
#### Examples 
##### C++ version (from scratch)
```bash
cd lab3/cpp
mkdir build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -mtune=native" .. && cmake --build .
```
pass parameters:
`path_to_input_image beta output_image`

run: 
```bash
./maxflow_ishikawa ../../test_images/input.png 15 out.png
```
##### Python (using PyMaxflow)
pass parameters:
`path_to_input_image beta n_labels output_image`

run: 
```bash
python3 py/ishikawa_maxflow.py ../test_images/input.png 15 5 out.png
```
