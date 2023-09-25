==============
Testing Assets
==============

Put all static file assets which you need for testing into this folder. This path can easily be accessed
via ``util.ASSETS_PATH`` in the ``tests`` folder.

What are some examples for testing assets?

* Maybe you want to test your processing of some Web API and don't want to rely on an internet connection
  for the tests? Simply export a few example queries as JSON and put them into the assets folder.
* You need to perform some hefty text processing? Add some raw text files to this folder
* You implement a complex algorithm with some big inputs or train an ML model? Add some mock datasets into
  this folder for the testing.
* ...


