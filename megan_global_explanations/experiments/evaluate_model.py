from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


# == EXPERIMENT PARAMETERS ==
# The parameters for the experiment.

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment
def experiment(e: Experiment):
    
    e.log('starting model evaluation...')
    
    # ~ load the dataset
    e.log('loading the dataset...')    
    
    # ~ load the model
    e.log('loading the model...')
    
    # ~ evaluating the model on the dataset
    e.log('starting model evaluation...')
    
    # Now we perform the clustering and the evaluation on that clustering.
    
    
experiment.run_if_main()
