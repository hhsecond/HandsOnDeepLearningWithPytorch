***************
Computer Vision
***************

Models
------
* Simple Convolutional Neural Networks
* Semantic Segmentation using Linknet 
* Capsule Networks

Datasets Used
-------------
I used different dataset for different models. ``Cfar-10`` has been used for simpleCNN which is part of ``torchvision`` models. The script automatically downloads the dataset. ``Camvid`` has been used for semantic segmentation which has been downloaded from `official Website`_ and kept it in `box`_ under the folder camvid

.. _box: https://app.box.com/s/25ict2irqaz3nnd19qp8ymtmkwx3l61j


Citations
---------
**Camvid**
Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008 (pdf)
Brostow, Shotton, Fauqueur, Cipolla (bibtex)

Semantic Object Classes in Video: A High-Definition Ground Truth Database (pdf)
Pattern Recognition Letters (to appear)
Brostow, Fauqueur, Cipolla (bibtex)

.. _official Website: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
