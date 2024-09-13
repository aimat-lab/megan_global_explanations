Changelog
=========

0.1.0 - 06.03.2024
------------------

Initial version

0.2.0 - 25.03.2024
------------------

- Added the function ``main.extract_concepts`` function which performs the concept extraction / clustering for a given 
  model and dataset combination and returns a list of all the identified concept dicts.
- Added the function ``generate_concept_prototypes`` which takes an existing list of concepts, the original model and the 
  dataset as parameters and will apply a genetic algorithm optimization to generate prototype graphs.

0.2.1 - 13.09.2024
------------------

- Modified the Reader and Writer classes to support the direct export and imports of the elements as visual graph elements 
  in the format of a visual graph dataset folder.