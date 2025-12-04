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

0.2.2 - 16.09.2024
------------------

- Fixed the Reader class to actually use the elements that were read with VisualGraphDatasetReader in the case of 
  explicitly passing the concepts.

0.3.0 - 04.12.2025
------------------

- Added the `concept_extraction.py` experiment which is a base experiment that does essentially the same 
  as the existing `vgd_concept_extraction.py` only that it does not need a pre-compiled visual graph dataset 
- Added `concept_extraction__aqsdolb.py`
- The concept extraction experiments now explicitly export the concept centroids in a JSON file.
- Refactored the `README.rst` file to now list the experiments based on the simple CSV files 
  first and only list the VGD based experiments as the second option