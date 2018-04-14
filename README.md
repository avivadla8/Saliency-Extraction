# 	Contrast Based Filtering for Salient Region Detection

## Main Paper:-
1. http://www.philkr.net/papers/2012-06-01-cvpr/2012-06-01-cvpr.pdf

## Additional Papers:
1. Saliency detection via graph-based manifold ranking, CVPR 2013
 * http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6619251

2. Saliency detection via dense and sparse reconstruction, CVPR 2013
 * http://ieeexplore.ieee.org/document/6751481/

3. Saliency-Aware Video Object Segmentation, 
 * http://ieeexplore.ieee.org/document/7837719/

4. Saliency-Aware Geodesic Video Object Segmentation
 * http://ieeexplore.ieee.org/document/7298961/

## Geodesic K-means Clustering:
* http://ieeexplore.ieee.org/document/4761241/

## Dataset:
* http://saliencydetection.net/dut-omron/

## Temporary links:
* http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2011/ACriminisi_ACM_TOG2010.pdf
* http://www.kev-smith.com/papers/SMITH_TPAMI12.pdf
* http://www.kev-smith.com/papers/SLIC_Superpixels.pdf
* http://davidstutz.de/wordpress/wp-content/uploads/2014/09/thesis.pdf
* https://in.mathworks.com/help/images/ref/imseggeodesic.html
* https://sites.google.com/site/uqchang/home/cv-code


## Usage:

* main.py --- It is implementation of initial Paper
 * usage : python main.py img_filename

* SaliencyFilter.py --- It is extension of Above paper where it is performed pixel wise
 * usage : python SaliencyFilter.py img_filename

* SaliencyOptimization.py --- It is implementation of Energy Minimization technique
 * usage : python SaliencyOptimization.py img_filename


