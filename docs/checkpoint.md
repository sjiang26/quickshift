# Checkpoint

## Updated Project Timeline
Saturday April 29th - finish testing sequential implementation

Tuesday May 2nd - finish implementing parallel implementation

Saturday May 6th - start implementing density approximation

Tuesday May 9th - finish implementing density approximation

Friday May 12th - finish testing and writeup, project due

## Work Completed So Far
Currently we have finished understanding and implementing the quick shift algorithm. We worked with learning how to convert images to pixel arrays, and switched from using MatLab, Octave, to openCV. We have also found a library of a sequential quickshift implementation, and we’re currently adding timing code to that we can use it to compare with our parallel implementation.

## Goals and Deliverables
With respect to our goals and deliverables, our timeline has slightly shifted as we did more research into the libraries and tools available to us. Originally, we were planning on developing a sequential implementation of the quick shift algorithm, but since we found an existing library of a sequential implementation, we’re going to build upon this implementation for testing and focus on parallelizing it for the GPU. We still plan on having a parallel implementation working for the competition, and we aim to use subsampling to further parallelize our algorithm.

## Parallelism Competition
At the parallelism competition, we plan to show a graph comparing the speedups of the sequential, parallel, and parallel with sub-sampling implementations. In addition, we plan on showing the before and after images of the image segmentation process.
