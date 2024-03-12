

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

Create the pipeline environment and install the cl_3d package
-------------------------------------------------------------

Setup and activate the included JSC environment (or use your own, e.g. Conda): 

.. code-block:: bash

   source environment/setup.sh
   source environment/activate.sh

Checkout `environment/README.md` for more info (e.g. on how to create a Jupyter kernel).

Before using the template, one needs to install the project as a package:

.. code-block:: bash

   pip install -e .

Versioneer
----------

This project uses `Versioneer <https://github.com/python-versioneer/python-versioneer>`_ to record package versions.

To create a new version use the `Git Tagging <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_ utility:

.. code-block:: bash

   git tag 1.2.3

To distribute it through gitlab push the tags and commits as

.. code-block:: bash

   git push; git push --tags

DataLad
-------

To use git annex in the local repository first run

.. code-block:: bash

   git-annex init

from its root.

To retrieve subdatasets run

.. code-block:: bash

   datalad get SUBMODULE

or

.. code-block:: bash

   datalad get --reckless=ephemeral SUBMODULE

if you just want to link to the data on a remote.
Additional sources of submodules are specified as `datalad.get.subdataset-source-candidate` in `.datalad/config` (See the `doc <http://handbook.datalad.org/en/latest/beyond_basics/101-148-clonepriority.html>`_).

cscratch
--------

Run

.. code-block:: bash

   ime-ctl -i --block -K data/subdataset/path/*

to make data available from cscratch

Remember to call

.. code-block:: bash

   export HDF5_USE_FILE_LOCKING='FALSE'

before using it

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


How to Cite
===========


If you use this work in your research, please cite it as follows:

.. code-block:: latex

   @article{oberstrass2024,
     title = {Self-{{Supervised Representation Learning}} for {{Nerve Fiber Distribution Patterns}} in {{3D-PLI}}},
     author = {Oberstrass, A. and others},
     year = {2024},
     journal = {arXiv preprint arXiv:2401.17207},
     eprint = {2401.17207},
     archiveprefix = {arxiv}
   }