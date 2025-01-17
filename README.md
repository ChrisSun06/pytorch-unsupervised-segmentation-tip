# Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering

This pytorch code generates segmentation labels of an input image.

![Unsupervised Image Segmentation with Scribbles](https://kanezaki.github.io/media/unsupervised_image_segmentation_with_scribbles.png)

Wonjik Kim\*, Asako Kanezaki\*, and Masayuki Tanaka.
**Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering.** 
*IEEE Transactions on Image Processing*, accepted, 2020.
([arXiv](https://arxiv.org/abs/2007.09990))

\*W. Kim and A. Kanezaki contributed equally to this work.

## What is new?

This is an extension of our [previous work](https://github.com/kanezaki/pytorch-unsupervised-segmentation). 

- Better performance with spatial continuity loss
- Option of using scribbles as user input
- Option of using reference image(s)

## Requirements

pytorch, opencv2, tqdm

## Getting started

### Vanilla

    $ python find_plant.py --input 0184_BP_H_-1_A_211_1595085939000_compressed.JPG

### Vanilla + scribbles

    $ python find_plant.py --input 0184_BP_H_-1_A_211_1595085939000_compressed.JPG --scribble

### Vanilla + reference image(s)

    $ python demo_ref.py --input ./BBC/
