import fiftyone as fo          # To use all the FiftyOne functionality
import fiftyone.zoo as foz     # To download custom dataset from Open Images Dataset
import os                      # To use operating system dependent functionality


# Check point
# Hint: to print emoji via Unicode, replace '+' with '000' and add prefix '\'
# For instance, emoji with Unicode 'U+1F44C' can be printed as '\U0001F44C'
print("Libraries are successfully loaded \U0001F44C")
print()

# Check point
# Showing currently active directory
print('Currently active directory is:')
print(os.getcwd())
print()


# Preparing directory to be activated
directory_to_save_dataset = os.path.join(os.getcwd(), "custom_dataset")


# Check point
print('The directory to be activated is:')
print(directory_to_save_dataset)
print()

# Activating needed directory
os.chdir(directory_to_save_dataset)

# Check point
print("The needed directory is successfully activated \U0001F44C")
print()

# Check point
# Showing currently active directory
print('Currently active directory is:')
print(os.getcwd())
print()


# Check point
# Showing current configuration
print(fo.config)
print()
print(fo.config.dataset_zoo_dir)
print()

# Assigning new path where to save downloaded dataset
fo.config.dataset_zoo_dir = directory_to_save_dataset


# Check point
# Showing current configuration
print(fo.config.dataset_zoo_dir)
print()


# Downloading custom dataset from Open Images Dataset
# Validation sub-dataset


# Setting number of samples
number_of_samples = 150


# All 3 classes together: cat, dog, elephant
dataset_validation = foz.load_zoo_dataset(
                                          "open-images-v6",
                                          split="validation",
                                          label_types=["detections"],
                                          classes=["Cat", "Dog", "Elephant"],
                                          max_samples=number_of_samples,
                                          shuffle=True,
                                          seed=51,
                                          dataset_name="cat-dog-elephant-150",
                                         )


print()

# Check point
# Showing dataset information
print(dataset_validation)
print()
print("-----")
print()


# By default, datasets are non-persistent
# Non-persistent datasets are deleted from the database each time the database is shut down
# However, the downloaded files on the disk are untouched


# Make the dataset persistent
dataset_validation.persistent = True


# Check point
# Showing dataset information
print(dataset_validation)
print()


# Loading again dataset into FiftyOne
dataset_validation = fo.load_dataset("cat-dog-elephant-150")


# Launching the App 1st time, and visualizing the dataset
# Creating session instance
session = fo.launch_app(dataset_validation)

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()

# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Loading again dataset into FiftyOne
dataset = fo.load_dataset("cat-dog-elephant-150")

# Deleting the dataset
dataset.delete()


# Check points
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())   # [] - list is empty
print(dataset.name)         # "cat-dog-elephant-150" - the name of the deleted dataset
print(dataset.deleted)      # if the dataset is deleted
print()


# (!) Pay attention
#     We deleted the FiftyOne reference only
#     The dataset itself (images and annotations) are saved in the hard drive


# Downloading custom dataset from Open Images Dataset
# Validation sub-dataset


# Setting number of samples
number_of_samples = 50


# All 3 classes separately: cat, dog, elephant
# Class: cat
dataset_validation = foz.load_zoo_dataset(
                                          "open-images-v6",
                                          split="validation",
                                          label_types=["detections"],
                                          classes=["Cat"],
                                          max_samples=number_of_samples,
                                          shuffle=True,
                                          seed=51,
                                          dataset_name="cat-dog-elephant-150-merged",
                                         )


# Class: dog
dog_subset = foz.load_zoo_dataset(
                                  "open-images-v6",
                                  split="validation",
                                  label_types=["detections"],
                                  classes=["Dog"],
                                  max_samples=number_of_samples,
                                  shuffle=True,
                                  seed=51,
                                  dataset_name="dog-50",
                                 )


# Class: elephant
elephant_subset = foz.load_zoo_dataset(
                                       "open-images-v6",
                                       split="validation",
                                       label_types=["detections"],
                                       classes=["Elephant"],
                                       max_samples=number_of_samples,
                                       shuffle=True,
                                       seed=51,
                                       dataset_name="elephant-50",
                                      )


print()


# Merging the samples together into the one dataset
_ = dataset_validation.merge_samples(dog_subset)
_ = dataset_validation.merge_samples(elephant_subset)


# Check point
print("The datasets are successfully merged \U0001F44C")
print()


# Loading again dataset into FiftyOne
dataset_validation = fo.load_dataset("cat-dog-elephant-150-merged")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_validation)


# Option 2
# Updating the session
session.dataset = dataset_validation


# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Iterating over all the datasets that are in FiftyOne
for dataset_name in fo.list_datasets():

    # Loading again dataset into FiftyOne
    dataset = fo.load_dataset(dataset_name)

    # Deleting the dataset
    dataset.delete()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())   # [] - list is empty
print()


# (!) Pay attention
#     We deleted the FiftyOne reference only
#     The dataset itself (images and annotations) are saved in the hard drive


# Downloading custom dataset from Open Images Dataset
# Validation dataset


# Setting number of samples
number_of_samples = 1000


# All 3 classes separately: cat, dog, elephant
# Class: cat
dataset_validation = foz.load_zoo_dataset(
                                          "open-images-v6",
                                          split="validation",
                                          label_types=["detections"],
                                          classes=["Cat"],
                                          max_samples=number_of_samples,
                                          seed=51,
                                          shuffle=True,
                                          dataset_name="cat-dog-elephant-validation",
                                         )


# Class: dog
dog_subset = foz.load_zoo_dataset(
                                  "open-images-v6",
                                  split="validation",
                                  label_types=["detections"],
                                  classes=["Dog"],
                                  max_samples=number_of_samples,
                                  seed=51,
                                  shuffle=True,
                                  dataset_name="dog-1000",
                                 )


# Class: elephant
elephant_subset = foz.load_zoo_dataset(
                                       "open-images-v6",
                                       split="validation",
                                       label_types=["detections"],
                                       classes=["Elephant"],
                                       max_samples=number_of_samples,
                                       seed=51,
                                       shuffle=True,
                                       dataset_name="elephant-1000",
                                      )


# Merging the samples together into the one dataset
_ = dataset_validation.merge_samples(dog_subset)
_ = dataset_validation.merge_samples(elephant_subset)


# Check point
print("The datasets are successfully merged \U0001F44C")
print()


# Loading again dataset into FiftyOne
dataset_validation = fo.load_dataset("cat-dog-elephant-validation")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_validation)


# Option 2
# Updating the session
session.dataset = dataset_validation


# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

# Loading again datasets into FiftyOne
dataset_dog = fo.load_dataset("dog-1000")
dataset_elephant = fo.load_dataset("elephant-1000")

# Deleting the datasets
dataset_dog.delete()
dataset_elephant.delete()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# (!) Pay attention
#     We deleted the FiftyOne reference only
#     The dataset itself (images and annotations) are saved in the hard drive

# Make the dataset persistent
dataset_validation.persistent = True


# Check point
# Showing dataset information
print(dataset_validation)
print()


# Downloading custom dataset from Open Images Dataset
# Test dataset


# Setting number of samples
number_of_samples = 1000


# All 3 classes separately: cat, dog, elephant
# Class: cat
dataset_test = foz.load_zoo_dataset(
                                    "open-images-v6",
                                    split="test",
                                    label_types=["detections"],
                                    classes=["Cat"],
                                    max_samples=number_of_samples,
                                    seed=51,
                                    shuffle=True,
                                    dataset_name="cat-dog-elephant-test",
                                   )


# Class: dog
dog_subset = foz.load_zoo_dataset(
                                  "open-images-v6",
                                  split="test",
                                  label_types=["detections"],
                                  classes=["Dog"],
                                  max_samples=number_of_samples,
                                  seed=51,
                                  shuffle=True,
                                  dataset_name="dog-1000",
                                 )


# Class: elephant
elephant_subset = foz.load_zoo_dataset(
                                       "open-images-v6",
                                       split="test",
                                       label_types=["detections"],
                                       classes=["Elephant"],
                                       max_samples=number_of_samples,
                                       seed=51,
                                       shuffle=True,
                                       dataset_name="elephant-1000",
                                      )


# Merging the samples together into the one dataset
_ = dataset_test.merge_samples(dog_subset)
_ = dataset_test.merge_samples(elephant_subset)


# Check point
print("The datasets are successfully merged \U0001F44C")
print()


# Loading again dataset into FiftyOne
dataset_test = fo.load_dataset("cat-dog-elephant-test")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_test)


# Option 2
# Updating the session
session.dataset = dataset_test

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

# Loading again dataset into FiftyOne
dataset_dog = fo.load_dataset("dog-1000")
dataset_elephant = fo.load_dataset("elephant-1000")

# Deleting the datasets
dataset_dog.delete()
dataset_elephant.delete()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

# Make the dataset persistent
dataset_test.persistent = True


# Check point
# Showing dataset information
print(dataset_test)
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Loading again dataset into FiftyOne
dataset_train = fo.load_dataset("cat-dog-elephant-train")
dataset_dog = fo.load_dataset("dog-2000")
dataset_elephant = fo.load_dataset("elephant-2000")

# Deleting the datasets
dataset_train.delete()
dataset_dog.delete()
dataset_elephant.delete()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Downloading custom dataset from Open Images Dataset
# Train dataset


# (!) In case of memory issue or Kernel stopped there are 2 options
#
#
#  Option 1 (to try to keep Kernel working):
#  1. Start from 500 samples
#  2. Run cell above to clear appropriate datasets' names
#  3. Start this cell again but now with 1000 samples
#  4. Run cell above to clear appropriate datasets' names
#  5. Start this cell again but now with 1500 samples
#  6. Run cell above to clear appropriate datasets' names
#  7. Start this cell again but now with 2000 samples
#
#
#  Option 2 (to restart Kernel each time it stops):
#  1. Start from maximum needed value, e.g. 2000
#  2. When Kernel stopped:
#     2.1 Restart Kernel
#     2.2 Repeat Step 1 (load libraries, change active directory, and set up path)
#  3. Run cell above to clear appropriate datasets' names
#  4. When Kernel stopped, repeat steps 2-3 until all samples are loaded


# Setting number of samples
number_of_samples = 2000    # or step by step: 500, 1000, 1500, and 2000


# All 3 classes separately: cat, dog, elephant
# Class: cat
dataset_train = foz.load_zoo_dataset(
                                     "open-images-v6",
                                     split="train",
                                     label_types=["detections"],
                                     classes=["Cat"],
                                     max_samples=number_of_samples,
                                     seed=51,
                                     shuffle=True,
                                     dataset_name="cat-dog-elephant-train",
                                    )


# Class: dog
dog_subset = foz.load_zoo_dataset(
                                  "open-images-v6",
                                  split="train",
                                  label_types=["detections"],
                                  classes=["Dog"],
                                  max_samples=number_of_samples,
                                  seed=51,
                                  shuffle=True,
                                  dataset_name="dog-2000",
                                 )


# Class: elephant
elephant_subset = foz.load_zoo_dataset(
                                       "open-images-v6",
                                       split="train",
                                       label_types=["detections"],
                                       classes=["Elephant"],
                                       max_samples=number_of_samples,
                                       seed=51,
                                       shuffle=True,
                                       dataset_name="elephant-2000",
                                      )


# Merging the samples together into the one dataset
_ = dataset_train.merge_samples(dog_subset)
_ = dataset_train.merge_samples(elephant_subset)


# Check point
print("The datasets are successfully merged \U0001F44C")
print()


# Loading again dataset into FiftyOne
dataset_train = fo.load_dataset("cat-dog-elephant-train")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_train)


# Option 2
# Updating the session
session.dataset = dataset_train

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Loading again dataset into FiftyOne
dataset_dog = fo.load_dataset("dog-2000")
dataset_elephant = fo.load_dataset("elephant-2000")

# Deleting the datasets
dataset_dog.delete()
dataset_elephant.delete()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Make the dataset persistent
dataset_train.persistent = True


# Check point
# Showing dataset information
print(dataset_train)
print()

# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# Loading datasets into FiftyOne
dataset_train = fo.load_dataset("cat-dog-elephant-train")
dataset_validation = fo.load_dataset("cat-dog-elephant-validation")
dataset_test = fo.load_dataset("cat-dog-elephant-test")


print("Datasets are successfully loaded \U0001F44C")
print()


# Check point
# Showing current configuration
print(fo.config.dataset_zoo_dir)
print()

# Preparing directory to export annotation file
directory_to_export_dataset = os.path.join(fo.config.dataset_zoo_dir, "open-images-v6")


# Check point
print('The directory to be activated is:')
print(directory_to_export_dataset)
print()


# Loading again dataset into FiftyOne
dataset_validation = fo.load_dataset("cat-dog-elephant-validation")


# The directory to which to write the exported labels
labels_path = os.path.join(directory_to_export_dataset, "validation", "export.json")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.COCODetectionDataset


# Export only labels of the detections in COCO format
dataset_validation.export(
                          labels_path=labels_path,
                          dataset_type=dataset_type,
                          label_field=label_field,
                         )

print()

# Loading again dataset into FiftyOne
dataset_test = fo.load_dataset("cat-dog-elephant-test")


# The directory to which to write the exported labels
labels_path = os.path.join(directory_to_export_dataset, "test", "export.json")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.COCODetectionDataset


# Export only labels of the detections in COCO format
dataset_test.export(
                    labels_path=labels_path,
                    dataset_type=dataset_type,
                    label_field=label_field,
                   )

print()


# Loading again dataset into FiftyOne
dataset_train = fo.load_dataset("cat-dog-elephant-train")


# The directory to which to write the exported labels
labels_path = os.path.join(directory_to_export_dataset, "train", "export.json")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.COCODetectionDataset


# Export only labels of the detections in COCO format
dataset_train.export(
                     labels_path=labels_path,
                     dataset_type=dataset_type,
                     label_field=label_field,
                    )

print()


# A name for the dataset
name = "cat-dog-elephant-validation-copy"


# The directory with dataset
dataset_dir = os.path.join(directory_to_export_dataset, "validation")


# The path to the COCO labels JSON file
labels_path = os.path.join(directory_to_export_dataset, "validation", "export.json")


# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset


dataset_validation_copy = fo.Dataset.from_dir(
                                              dataset_dir=dataset_dir,
                                              dataset_type=dataset_type,
                                              labels_path=labels_path,
                                              name=name,
                                             )

print()

# Check point
# Showing dataset information
print(dataset_validation_copy)
print()


# Loading again dataset into FiftyOne
dataset_validation_copy = fo.load_dataset("cat-dog-elephant-validation-copy")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_validation_copy)


# Option 2
# Updating the session
session.dataset = dataset_validation_copy

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()

# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# A name for the dataset
name = "cat-dog-elephant-test-copy"


# The directory with dataset
dataset_dir = os.path.join(directory_to_export_dataset, "test")


# The path to the COCO labels JSON file
labels_path = os.path.join(directory_to_export_dataset, "test", "export.json")


# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset


dataset_test_copy = fo.Dataset.from_dir(
                                        dataset_dir=dataset_dir,
                                        dataset_type=dataset_type,
                                        labels_path=labels_path,
                                        name=name,
                                       )

print()


# Check point
# Showing dataset information
print(dataset_test_copy)
print()

# Loading again dataset into FiftyOne
dataset_test_copy = fo.load_dataset("cat-dog-elephant-test-copy")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_test_copy)


# Option 2
# Updating the session
session.dataset = dataset_test_copy

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()


# A name for the dataset
name = "cat-dog-elephant-train-copy"


# The directory with dataset
dataset_dir = os.path.join(directory_to_export_dataset, "train")


# The path to the COCO labels JSON file
labels_path = os.path.join(directory_to_export_dataset, "train", "export.json")


# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset


dataset_train_copy = fo.Dataset.from_dir(
                                         dataset_dir=dataset_dir,
                                         dataset_type=dataset_type,
                                         labels_path=labels_path,
                                         name=name,
                                        )

print()


# Check point
# Showing dataset information
print(dataset_train_copy)
print()


# Loading again dataset into FiftyOne
dataset_train_copy = fo.load_dataset("cat-dog-elephant-train-copy")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_train_copy)


# Option 2
# Updating the session
session.dataset = dataset_train_copy


# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()


# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

