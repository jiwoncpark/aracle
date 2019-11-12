======
aracle
======

.. image:: https://travis-ci.com/jiwoncpark/aracle.svg?branch=master
    :target: https://travis-ci.org/jiwoncpark/aracle

.. image:: https://readthedocs.org/projects/aracle/badge/?version=latest
        :target: https://aracle.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/jiwoncpark/aracle/badge.svg?branch=master
        :target: https://coveralls.io/github/jiwoncpark/aracle?branch=master


Prediction of solar magnetic flux emergence using deep neural nets. "Active region (AR) oracle"

Installation
============

1. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it:

::

$conda create -n aracle python=3.6 -y
$conda activate aracle

2. Now do one of the following. 

**Option 2(a):** clone the repo (please do this if you'd like to contribute to the development).

::

$git clone https://github.com/jiwoncpark/aracle.git
$cd aracle
$pip install -e . -r requirements.txt

**Option 2(b):** pip install the release version (only recommended if you do not plan to contribute to the development).

::

$pip install aracle


3. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name aracle --display-name "Python (aracle)"

How to train
============

1. Generate the training toy data, e.g.

::

$python -m aracle.toy_data.generate_toy_data 5 224 ./my_data 

2. Run

::

$python -m aracle.train_faster_rcnn

You can visualize the training results by running

::

$tensorboard --logdir runs

Feedback and More
=================

Suggestions are always welcome! If you encounter issues or areas for improvement, please message @jiwoncpark or `make an issue
<https://github.com/jiwoncpark/aracle/issues>`_.