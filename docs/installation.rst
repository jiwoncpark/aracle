============
Installation
============

1. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it:

::

$conda create -n aracle python=3.6 -y
$conda activate aracle

2. Install PyTorch stable and torchvision following the [official instructions](https://pytorch.org/), e.g.,

::

$conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

3. Now do one of the following. 

**Option 3(a):** clone the repo (please do this if you'd like to contribute to the development).

::

$git clone https://github.com/jiwoncpark/aracle.git
$cd aracle
$pip install -e .

**Option 3(b):** pip install the release version (only recommended if you're a user). Not available yet!

::

$pip install aracle


4. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name aracle --display-name "Python (aracle)"