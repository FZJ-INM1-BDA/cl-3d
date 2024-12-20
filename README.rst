
.. image:: https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white
    :alt: PyTorch
    :target: https://pytorch.org/get-started/locally/

.. image:: https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white
    :alt: Lightning
    :target: https://pytorchlightning.ai/

.. image:: https://img.shields.io/badge/Config-Hydra-89b8cd
    :alt: Config: Hydra
    :target: https://hydra.cc/

.. image:: https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray
    :alt: Lightning-Hydra-Template
    :target: https://github.com/HelmholtzAI-Consultants-Munich/ML-Pipeline-Template

.. image:: https://img.shields.io/badge/-Pyscaffold--Datascience-017F2F?style=flat&logo=github&labelColor=gray
    :alt: Pyscaffold-Datascience
    :target: https://github.com/pyscaffold/pyscaffoldext-dsproject

|

=====
CL-3D
=====

Contrastive learning using 3D context of stacked brain sections.

Quickstart
==========


Install the cl_3d package
-------------------------

Clone the `cl-3d` repository and install it as editable packge:

.. code-block:: bash

    git clone https://jugit.fz-juelich.de/inm-1/bda/personal/aoberstrass/projects/cl-3d.git
    cd cl-3d
    pip install -e .

The installation requires Python 3.10 and the GCC, OpenMPI and LibTIFF packages.

On Ubuntu you can install the packages as:

.. code-block:: bash

    sudo apt install build-essential libtiff-dev libopenmpi-dev

install them using conda:

.. code-block:: bash

    conda install gxx_linux-64==9.3.0 anaconda::openmpi conda-forge::pylibtiff

or load the corresponding modules when working on the Jülich Supercomputing facility:

.. code-block:: bash
    
   ml Stages/2024
   ml load GCC/12.3.0 OpenMPI/4.1.5 LibTIFF/.4.5.0


Training
--------

For local debugging (not on a compute node) start the script as

.. code-block:: bash
    
    HYDRA_FULL_ERROR=1 python scripts/train.py debug=step

or submit a job to the development partition on the JSC:

.. code-block:: bash
    
    sbatch scripts/dev_jureca.sbatch default

For full-scale training submit the job as

.. code-block:: bash
    
    sbatch scripts/train_jureca.sbatch default

In both cases `default` can be replaced with an experiment configuration under `configs/experiment`.


Inference
---------

To apply a trained model to unseen sections use `python scripts/apply-encoder.py` with a path to the checkpoint of a trained model for `--ckpt`.


Project Organization
====================

::

    ├── configs                              <- Hydra configuration files
    │   ├── callbacks                               <- Callbacks configs
    │   ├── datamodule                              <- Datamodule configs
    │   ├── debug                                   <- Debugging configs
    │   ├── experiment                              <- Experiment configs
    │   ├── hparams_search                          <- Hyperparameter search configs
    │   ├── local                                   <- Local configs
    │   ├── log_dir                                 <- Logging directory configs
    │   ├── logger                                  <- Logger configs
    │   ├── model                                   <- Model configs
    │   ├── trainer                                 <- Trainer configs
    │   │
    │   ├── test.yaml                               <- Main config for testing
    │   └── train.yaml                              <- Main config for training
    │
    ├── environment                          <- Computing environment
    │   ├── requirements                            <- Python packages and JSC modules requirements
    │   │
    │   ├── activate.sh                             <- Activation script
    │   ├── config.sh                               <- Environment configurations  
    │   ├── create_kernel.sh                        <- Jupyter Kernel script
    │   └── setup.sh                                <- Environment setup script
    │
    ├── logs
    │   ├── experiments                      <- Logs from experiments
    │   ├── slurm                            <- Slurm outputs and errors
    │   └── tensorboard/mlruns/...           <- Training monitoring logs
    |
    ├── models                               <- Trained and serialized models, model predictions
    |
    ├── notebooks                            <- Jupyter notebooks
    |
    ├── scripts                              <- Scripts used in project
    │   ├── train_juwels.sbatch                     <- Submit job to slurm on JUWELS
    │   ├── test.py                                 <- Run testing
    │   └── train.py                                <- Run training
    │
    ├── src/cl_3d                            <- Source code
    │   ├── datamodules                             <- Lightning datamodules
    │   ├── models                                  <- Lightning models
    │   ├── utils                                   <- Utility scripts
    │   │
    │   ├── testing_pipeline.py
    │   └── training_pipeline.py
    │
    ├── .coveragerc                          <- Configuration for coverage reports of unit tests.
    ├── .gitignore                           <- List of files/folders ignored by git
    ├── .pre-commit-config.yaml              <- Configuration of pre-commit hooks for code formatting
    ├── setup.cfg                            <- Configuration of linters and pytest
    ├── LICENSE.txt                          <- License as chosen on the command-line.
    ├── pyproject.toml                       <- Build configuration. Don't change! Use `pip install -e .`
    │                                           to install for development or to build `tox -e build`.
    ├── setup.cfg                            <- Declarative configuration of your project.
    ├── setup.py                             <- [DEPRECATED] Use `python setup.py develop` to install for
    │                                           development or `python setup.py bdist_wheel` to build.
    └── README.md


DataLad
-------

To retrieve the training data run

.. code-block:: bash

   datalad get datasets/vervet1818-3d-pairs/

or

.. code-block:: bash

   datalad get --reckless=ephemeral datasets/vervet1818-3d-pairs

if you just want to link to the data on a remote without copying the files.
Additional sources of submodules are specified as `datalad.get.subdataset-source-candidate` in `.datalad/config` (See the `doc <http://handbook.datalad.org/en/latest/beyond_basics/101-148-clonepriority.html>`_).

Please note that access to the data can only be provided on request.


cscratch
--------

To use `cscratch` on JSC run

.. code-block:: bash

   ime-ctl -i --block -K data/subdataset/path/*

to make data available from cscratch and

.. code-block:: bash

   export HDF5_USE_FILE_LOCKING='FALSE'

to disable file locking.


How to Cite
===========

When using this work in your research, please cite it as follows:

.. code-block:: latex

  @article{oberstrass2024,
    title = {Self-{{Supervised Representation Learning}} for {{Nerve Fiber Distribution Patterns}} in {{3D-PLI}}},
    author = {Oberstrass, Alexander and Muenzing, Sascha E. A. and Niu, Meiqi and {Palomero-Gallagher}, Nicola and Schiffer, Christian and Axer, Markus and Amunts, Katrin and Dickscheid, Timo},
    year = {2024},
    journal = {Imaging Neuroscience},
    issn = {2837-6056},
    doi = {https://doi.org/10.1162/imag_a_00351},
  }
