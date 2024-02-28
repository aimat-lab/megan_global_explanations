import typing as t

# ~ Typing
# Here we are setting up some typing aliases for convenience.

"""
:type ConceptDict: 

    This type describes a special kind of dictionary structure that encodes all the 
    information about a single global concept explanation. This dictionary is used as the standard 
    transfer format of the concept data.

    The concept dictionary is a dynamic data structure, which consists of only a few required fields
    and a number of optional fields. The required fields are:
    - centroid: The centroid of the concept in the model's embedding space.
    - num: The number of concepts associated with the prototype.
"""
ConceptDict = t.Dict[str, t.Any]

"""
:type ConceptData: 

    This type describes a list of the special concept dictionaries. As a list of 
    multiple concepts, this data structure represents the complete global concept explanations for 
    a single model and dataset combination.

    In other words, this data structure is the used to represent the result of a complete global 
    explanation method.
"""
ConceptData = t.List[ConceptDict]