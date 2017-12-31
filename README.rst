**************
ThePyTorchBook
**************

Code used for the book
======================

Repository is arranged chapter wise and each folder includes the code used + the visualization of models used. Visualization is created by using `Lucent`_. Dataset used for the models are either available in the shared `box`_ folder or downloadable from the ``torch`` utility packages such as ``torchvision``, ``torchtext`` or ``torchaudio``

.. _box: https://app.box.com/s/25ict2irqaz3nnd19qp8ymtmkwx3l61j

.. _Lucent: https://github.com/hhsecond/lucent

Chapters
--------
#. Introduction
#. A Simple Neural Network
#. Nuts And Bolts
#. Computer Vision
#. Sequential Data Processing
#. Generative Networks
#. Reinforcement Learning
#. PyTorch In Production


Utilities
---------
* Visualization is handled by Lucent - ::

    pip install lucent
* Environment is handled by Pipenv - ::

    pip install pipenv

Usage
-----
* Clone the repository ::

    git clone https://github.com/hhsecond/ThePyTorchBook.git && cd ThePyTorchBook

* Install dependancies. ThePyTorchBook is using python3.6 and pipenv for package management ::

    pipenv install

* CD to chapter directores and execute the models


