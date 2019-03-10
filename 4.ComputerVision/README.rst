***************
Computer Vision
***************
(TODO Perhaps add attention)

Models
------
* Simple Convolutional Neural Networks
* Semantic Segmentation using Linknet 
* Capsule Networks

Datasets Used
-------------
I used different dataset for different models. ``Cifar-10`` has been used for simpleCNN which is part of ``torchvision`` models. The script automatically downloads the dataset. ``Camvid`` has been used for semantic segmentation which has been downloaded from `official Website`_ and kept it in `box`_ under the folder camvid

.. _box: https://app.box.com/s/25ict2irqaz3nnd19qp8ymtmkwx3l61j


Citations
---------
**Camvid** `Bibtex`_ (Other links seem to be broken)

.. _Bibtex: http://www0.cs.ucl.ac.uk/staff/G.Brostow/bibs/RecognitionFromMotion_bib.html

(1)
Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008
Brostow, Shotton, Fauqueur, Cipolla

(2)
Semantic Object Classes in Video: A High-Definition Ground Truth Database
Pattern Recognition Letters
Brostow, Fauqueur, Cipolla

.. _official Website: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
