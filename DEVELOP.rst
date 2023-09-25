======================
Development Cheatsheet
======================

Poetry
======

Setting up virtualenv development environment
---------------------------------------------

This project uses Poetry_ for package managment

First you need to create a new ``virtualenv`` in the root directory of the project. Then you need to
activate that environment and install Poetry_ into it.

.. code-block:: shell

    python3 -m venv ./
    ./venv/bin/activate
    pip3 install poetry

Then you can use ``poetry install`` to automatically install all the package dependencies listed within the
``pyproject.toml`` file.

.. code-block:: shell

    python3 -m poetry install
    python3 -m poetry env use ./venv/bin/python

**NOTE:** Whenever invoking any poetry command within the virtualenv it is
*necessary* to use the the format ``python -m poetry`` instead of just ``poetry`` because the latter will
always attempt to use the system python binary and not the venv binary!

.. _Poetry: https://python-poetry.org/


Git
===

Add Remote Repository
---------------------

It makes sense to directly supply a Github personal auth token when registering a new remote location for
the local repository, because that will remove any hassle with authentication when trying to push in the
future.

.. code-block:: shell

    git remote add origin https:://[github_username]:[github_token]@github.com/the16thpythonist/megan_global_explanations.git
    git push origin master


Create Anonymous Github Repository
----------------------------------

Some journals / conferences use a double blind review process, which means that all aspects of a submission
need to be anonymized. This also includes the code that is submitted alongside the paper. This project
already implements the ``anon`` and ``de-anon`` commands which can be used to replace potentially identifying
information about the authors with random hashes.

But beyond the contents of the repository, the repository itself needs to be anonymous. This means you have
to create a new github account and create a new repository there.

To do this you can follow these steps:

**(1)** Create a `new gmail account`_

**(2)** Use that to create a `new github account`_ as well

**(3)** Make sure to retrieve a personal access token for that account from the github developer settings

**(4)** Setup a new remote location for your local repository

.. code-block:: shell

    git remote add anon https://[username]:[access_token]@github.com/[username]/megan_global_explanations.git

**(5)** Create a new *orphan* branch and push to the repo

.. code-block:: shell

    git checkout -b --orphan anon
    git commit -a -m "anon"
    git push anon anon


.. _new gmail account: https://accounts.google.com/signup/v2/webcreateaccount?flowName=GlifWebSignIn&flowEntry=SignUp
.. _new github account: https://github.com/join
