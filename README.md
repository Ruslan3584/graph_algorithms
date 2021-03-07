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
path_to_input_image binary_penalty output_image

run: 
```bash
./maxflow ../test_images/input.png 50 ../test_images/result.png
```
