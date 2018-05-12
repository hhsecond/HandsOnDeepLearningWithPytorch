******************************
HandsOnDeepLearningWithPytorch
******************************
[Version 1.0 to be out by August '18]
"""""""""""""""""""""""""""""""""""""

Code used for the book
======================

Repository is arranged chapter wise and each folder includes the code used + the visualization of models used. One of the greatest introduction of the book is the AI exploration platform `TuringNetwork`_. Dataset used for the models are either available in the shared `box`_ folder or downloadable from the ``torch`` utility packages such as ``torchvision``, ``torchtext`` or ``torchaudio``

.. _box: https://app.box.com/s/25ict2irqaz3nnd19qp8ymtmkwx3l61j

.. _TuringNetwork: https://github.com/dlguys/flashlight

Chapters
--------
#. Introduction
#. A Simple Neural Network
#. Deep Learning work flow
#. Computer Vision
#. Sequential Data Processing
#. Generative Networks
#. Reinforcement Learning
#. PyTorch In Production


Utilities
---------
* Visualization is handled by TuringNetwork - ::

    pip install turingnetwork
* Environment is handled by Pipenv - ::

    pip install pipenv

Usage
-----
* Clone the repository ::

    git clone https://github.com/hhsecond/HandsOnDeepLearningWithPytorch.git && cd HandsOnDeepLearningWithPytorch

* Install dependancies. HandsOnDeepLearningWithPytorch is using python3.7 and pipenv for package management ::

    pipenv install

* CD to chapter directores and execute the models


