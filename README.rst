.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==================
Self-Supervised Learning for Remote sensing
==================


Researching the effect of different self-supervised pre-text tasks on downstream task performance.

Description
-----------
In this repository we implement three different pre-text tasks for self-supervised learning on satellite imagery
and compare model performance after fine-tuning on two different downstream tasks (namely, semantic segmentation
and classification). We use a ResNet-18 as the Encoder and mirror it in a U-Net-esque architecture
for semantic segmentation while we use a simple fully connected neural network for the classification task.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
