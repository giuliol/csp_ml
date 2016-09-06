from src.test_units import Tests
from src.tools.dataset_helper import DatasetHelper
# Tests.tools_test()
# Tests.tflearn_autoencoder.tflearn_tutorial()

training_set = DatasetHelper.load_data("res/dummy_set/training")
test_set = DatasetHelper.load_data("res/dummy_set/test")

print(training_set.shape)
print(test_set.shape)



# Tests.giulio_test()