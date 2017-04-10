## Summary
We are going to implement image segmentation on a GPU using parallelized quick shift, a fast mode seeking algorithm.

## Background
Image segmentation is the process of dividing an image into multiple segments, in order to change the image into something easier to analyze. Segmentation algorithms have played a big role in computer vision research, and can be implemented using many different algorithms, some of which include thresholding, clustering, using minimum spanning trees, and quick shift. In this project, we will explore the implementations of quick shift for image segmentation and its optimizations using parallelization in CUDA and openMP. Quick shift is a kernelized version of a mode seeking algorithm which is similar conceptually to mean shift or medoid shift. 

The main idea of quick shift is to move each point to the nearest neighbor that would lead to an increment of the density, and the calculations of the density follow from similar equations for mean shift or medoid shift. All the pixels and its connections with its neighbors form a tree, where the root of the tree is the point with the highest density estimate. Each connection between points has an associated distance, and the segmentation of the tree is calculated by removing all links in the tree that have a distance that’s greater than a chosen threshold. Each segment is formed by the pixels that are a part of each resulting disconnected tree.

The following is pseudocode for quick shift image segmentation:
```
function computeDensity()
for x in all pixels
  P[x] = 0
  for n in all pixels less than 3*sigma away
    P[x] += exp(-(f[x]-f[n])^2 / (2*sigma*sigma))
    
function linkNeighbors()
for x in all pixels
  for n in all pixels less than tau away
    if P[n] > P[x] and distance(x,n) is smallest among all n
      d[x] = distance(x,n)
      parent[x] = n
```

Quick shift image segmentation is prime for parallelization because density computations and neighbor calculations occur on each pixel of the image, and the computations don’t affect the pixels are distant from one another. This way, we can parallelize across pixels.

## Challenge
Implementing the sequential algorithm described in the paper in parallel will be challening. Additionally, the bottleneck on this algorithm is memory latency, since global memory on the GPU is slow. In the paper, they address this by using a texture cached approach of loading pixels. This will be tricky to implement.

## Resources
We plan to start from the methods described in this paper:
["Brian Fulkerson and Stefano Soatto, et. al Really quick shift: Image segmentation on a GPU"](http://www.vision.cs.ucla.edu/papers/fulkersonS10really.pdf).
We will also be using NVIDIA GeForce GTX 1080 on the GHC machines.

## Goals and Deliverables
#### Plan to achieve
Implement the quick shift algorithm as described in the paper and obtain similar speedup (10-50x speedup on practical images) over the CPU implementation.

#### Hope to achieve
Try to achieve higher speedup by approximating density via subsampling.

#### Demo
In our demo, we plan to show the difference between the CPU implementation and our implementation in regards to time, but the similarity in quality of segmentation.

## Platform Choice
Similar to assignment 2, we will be writing in C++ and using CUDA to implement the parallelized quick shift algorithm on the GPU. We will be running the code on the NVIDIA GeForce GTX 1080 on the GHC machines. These computers would be useful for this implementation because its architecture allows us to test on multi-core machines that employs CUDA processing.

## Schedule
April 12 - Finish researching and understanding the algorithm
April 15 - Have sequential implementation of quick shift image segmentation
April 25 (Checkpoint) - Have major CUDA kernels written
April 29 - Finish CUDA implementation
May 6 - Improvements and testing
May 12 - Final Presentation
