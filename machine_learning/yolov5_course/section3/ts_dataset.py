# Installing 'tqdm' library to show smart progress bars
# and track calculations inside loops in Real Time
# Adding flag '-y' for silent installation
# !conda install -c anaconda tqdm -y
# pip3 install tqdm
print('Installed tqdm \U0001F44C')

import fiftyone as fo                 # To use all the FiftyOne functionality
import fiftyone.utils.random as fous  # To split dataset into sub-datasets
import os                             # To use operating system dependent functionality
import glob                           # To find all the pathnames matching a specified pattern
import pandas as pd                   # To load and process dataFrames
import cv2                            # To load and process images
from tqdm import tqdm                 # To track calculations inside loops in Real Time

# Check point
# Hint: to print emoji via Unicode, replace '+' with '000' and add prefix '\'
# For instance, emoji with Unicode 'U+1F44C' can be printed as '\U0001F44C'
print("Libraries are successfully loaded \U0001F44C")
print()

# Verifying successfull 'tqdm' installation
for i in tqdm(range(10000000)):
    ...
print('tqdm installed ...')

# Check point
# Showing currently active directory
print('Currently active directory is:')
print(os.getcwd())
print()

# Preparing paths to directories
directory_ts_original = os.path.join(os.getcwd(), "ts_original")
directory_ts_yolo_4_classes = os.path.join(os.getcwd(), "ts_yolo", "yolov5dataset", "ts4classes")
directory_ts_yolo_43_classes = os.path.join(os.getcwd(), "ts_yolo", "yolov5dataset", "ts43classes")


# Check point
print('The paths to directories are:')
print(directory_ts_original)
print(directory_ts_yolo_4_classes)
print(directory_ts_yolo_43_classes)
print()


# Defining lists of classes according to the classes ID's


# Prohibitory group:
# Circular traffic signs with white background and red border line
prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]


# Danger group:
# Triangular traffic signs with white background and red border line
danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


# Mandatory group:
# Circular traffic signs with blue background
mandatory = [33, 34, 35, 36, 37, 38, 39, 40]


# Other group:
other = [6, 12, 13, 14, 17, 32, 41, 42]


# Check point
print("The groups of classes are successfully prepared \U0001F44C")
print()


# Reading txt file with annotations separated by semicolon
# Loading 6 columns into Pandas dataFrame
# Giving at the same time names to the columns
annotations = pd.read_csv(os.path.join(directory_ts_original, "gt.txt"),
                          names=['ImageID',
                                 'XMin',
                                 'YMin',
                                 'XMax',
                                 'YMax',
                                 'ClassID'],
                          sep=';')


# Check point
print("The annotations are successfully loaded \U0001F44C")
print()

# Check point
# Showing first 5 rows from the dataFrame
print(annotations.head())

# Calculating bounding boxes' width and height for all rows at the same time
annotations['XMax'] = annotations['XMax'] - annotations['XMin']  # width of the bounding box
annotations['YMax'] = annotations['YMax'] - annotations['YMin']  # height of the bounding box

# Renaming columns of the dataFrame inplace
annotations.rename(columns={'XMax': 'Width', 'YMax': 'Height'}, inplace=True)

# Check point
print("The width and height are successfully calculated \U0001F44C")
print("The appropriate columns are successfully renamed \U0001F44C")
print()

# Check point
# Showing first 5 rows from the dataFrame
print(annotations.head())
print()

# Preparing path pattern to find images
# images_patt = directory_ts_original + "/*.ppm"
images_patt = os.path.join(directory_ts_original, '*.ppm')

# Iterating all the found images
for filepath in tqdm(glob.glob(images_patt)):
    # Reading current image and getting its real width and height
    ppm_image = cv2.imread(filepath)
    
    # Slicing from tuple only first two elements
    # We need real image height and width to normalize bounding boxes' coordinates
    height_image, width_image = ppm_image.shape[:2]
    
    # Slicing only image filename and name
    # filepath         --> C:\Users\valen\yolov5course\section3\ts_original\00000.ppm
    # filapath[-9:]    --> 00000.ppm
    # filapath[-9:-4]  --> 00000
    filename_image = filepath[-9:]
    name_image = filepath[-9:-4]
    
    # Preparing path where to save JPG image
    path_to_jpg_image = os.path.join(directory_ts_original, name_image) + '.jpg'
    
    # Saving image in JPG format by OpenCV
    # OpenCV uses extension to choose format to save image with
    cv2.imwrite(path_to_jpg_image, ppm_image)
    
    # Getting sub-dataFrame
    # Locating needed row(s) in the dataFrame for current image
    # By using 'loc' method and condition 'annotations['ImageID'] == filename_image'
    # we find row(s) with annotations for current image
    # By using 'copy()' we create separate sub-dataFrame
    rows_image = annotations.loc[annotations['ImageID'] == filename_image].copy()
    
    # Checking if there is no any annotations for current image
    if rows_image.isnull().values.all():
        # Skipping this image
        continue
    
    # Normalizing bounding boxes' coordinates
    # according to the real image width and height
    rows_image["XMin"] = rows_image["XMin"] / width_image
    rows_image["YMin"] = rows_image["YMin"] / height_image
    rows_image["Width"] = rows_image["Width"] / width_image
    rows_image["Height"] = rows_image["Height"] / height_image
    
    # Updating dataFrame with normalized values for current image
    annotations.loc[annotations['ImageID'] == filename_image] = rows_image
    
print()
# Check point
print("Conversion and normalization are successfully done \U0001F44C")
print()

# Check point
# Showing first 5 rows from the dataFrame
print(annotations.head())
print()


# Check point
# Showing list of loaded datasets into FiftyOne
# next lines failed many times.

try:
    print(fo.list_datasets())
except:
    print()


# Deleting datasets from FiftyOne
try:
    fo.load_dataset("ts-empty-classes-yolov5").delete()
except:
    print()

try:
    fo.load_dataset("ts-4-classes-yolov5").delete()
except:
    print()

try:
    fo.load_dataset("ts-43-classes-yolov5").delete()
except:
    print()

try:
    fo.load_dataset("ts-4-classes-yolov5-copy").delete()
except:
    print()

try:
    fo.load_dataset("ts-43-classes-yolov5-copy").delete()
except:
    print()


# Preparing path pattern to find images
images_patt = directory_ts_original + "/*.jpg"


# Preparing lists to collect image samples in
samples_ts_4 = []      # for image samples that have traffic signs
samples_ts_43 = []     # for image samples that have traffic signs
samples_ts_empty = []  # for image samples that does not have traffic signs
# Images without traffic signs will be used by YOLO v5, when training


# Iterating all the found images
for filepath in glob.glob(images_patt):
    
    # Slicing only image name
    # filepath         --> C:\Users\valen\yolov5course\section3\ts_original\00000.jpg
    # filapath[-9:-4]  --> 00000
    name_image = filepath[-9:-4]
    
    
    # Getting sub-dataFrame
    # Locating needed row(s) in the dataFrame for current image
    # By using 'loc' method and condition 'annotations['ImageID'] == filename_image'
    # we find row(s) with annotations for current image
    # By using 'copy()' we create separate sub-dataFrame
    rows_image = annotations.loc[annotations['ImageID'] == name_image + ".ppm"].copy()
    
    
    # Preparing lists to collect detections for current image in
    detections_ts_4 = []      # for images that have traffic signs
    detections_ts_43 = []     # for images that have traffic signs
    detections_ts_empty = []  # for images that does not have traffic signs
    
    
    # Checking if there is any annotation for current image
    if rows_image.isnull().values.all():  # True, no annotations for current image
        
        # Preparing empty class
        label = ""
        
        
        # Preparing empty bounding box coordinates
        bounding_box = []
        
        
        # Adding empty annotation
        detections_ts_empty.append(fo.Detection(label=label, bounding_box=bounding_box))
        
        
        # Getting current image sample
        sample_empty = fo.Sample(filepath=filepath)
        
        
        # print(sample_empty)
        #
        # <Sample: {
        #          'id': None,
        #          'media_type': 'image',
        #          'filepath': 'C:\\Users\\valen\\yolov5course\\section3\\ts_original\\00000.jpg',
        #          'tags': [],
        #          'metadata': None
        #         }>
        
        
        # Storing epmty detections in a field 'detections'
        # Referencing all detections to current image sample
        sample_empty["detections"] = fo.Detections(detections=detections_ts_empty)
        
        
        # Adding current image sample with empty detections to the list of samples
        samples_ts_empty.append(sample_empty)
        
    
    else:  # False, there is(are) annotation(s) for current image
        
        # Iterating all found objects (rows) for current image        
        for index, row in rows_image.iterrows():
            
            # Preparing current class for TS dataset with 4 classes
            if row["ClassID"] in prohibitory:
                label_4 = "0"        # [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
            
            elif row["ClassID"] in danger:
                label_4 = "1"        # [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            
            elif row["ClassID"] in mandatory:
                label_4 = "2"        # [33, 34, 35, 36, 37, 38, 39, 40]
            
            elif row["ClassID"] in other:
                label_4 = "3"        # [6, 12, 13, 14, 17, 32, 41, 42]
            
            
            # Preparing current class for TS dataset with 43 classes
            # Converting the integer class id to the string by 'str'
            label_43 = str(row["ClassID"])
            
            
            # Preparing current bounding box coordinates
            # Bounding box coordinates should be relative values
            # in [0, 1] in the following format:
            # [top-left-x, top-left-y, width, height]
            bounding_box = [row["XMin"], row["YMin"], row["Width"], row["Height"]]
            
            
            # Adding current object annotation to the list of detections for current image
            detections_ts_4.append(fo.Detection(label=label_4, bounding_box=bounding_box))
            detections_ts_43.append(fo.Detection(label=label_43, bounding_box=bounding_box))
            
            
            # print(detections_ts_43)
            #
            # [<Detection: {
            #               'id': '62e4ffef6cafa3f389dd2f4d',
            #               'attributes': BaseDict({}),
            #               'tags': BaseList([]),
            #               'label': '11',
            #               'bounding_box': BaseList([0.569118, 0.51375, 0.030147, 0.04375]),
            #               'mask': None,
            #               'confidence': None,
            #               'index': None,
            #        }>]
        
        
        # Getting current image sample
        sample_4 = fo.Sample(filepath=filepath)
        sample_43 = fo.Sample(filepath=filepath)
        
        
        # Storing detections in a field 'detections'
        # Referencing all detections to current image sample
        sample_4["detections"] = fo.Detections(detections=detections_ts_4)
        sample_43["detections"] = fo.Detections(detections=detections_ts_43)
        
        
        # Adding current image sample with all its detections to the list of samples
        samples_ts_4.append(sample_4)
        samples_ts_43.append(sample_43)
    

# Creating datasets with empty detections
dataset_ts_empty = fo.Dataset("ts-empty-classes-yolov5")
dataset_ts_empty.add_samples(samples_ts_empty)


# Creating datasets with 4 classes
dataset_ts_4 = fo.Dataset("ts-4-classes-yolov5")
dataset_ts_4.add_samples(samples_ts_4)

# Creating datasets with 43 classes
dataset_ts_43 = fo.Dataset("ts-43-classes-yolov5")
dataset_ts_43.add_samples(samples_ts_43)

print()

# Check point
print("Datasets are successfully created \U0001F44C")
print()

# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

# Loading again datasets into FiftyOne
dataset_ts_empty = fo.load_dataset("ts-empty-classes-yolov5")
dataset_ts_4_classes = fo.load_dataset("ts-4-classes-yolov5")
dataset_ts_43_classes = fo.load_dataset("ts-43-classes-yolov5")


# Making the datasets persistent
dataset_ts_empty.persistent = True
dataset_ts_4_classes.persistent = True
dataset_ts_43_classes.persistent = True


# Check point
# Showing dataset information
print(dataset_ts_empty)
print("---")
print(dataset_ts_4_classes)
print("---")
print(dataset_ts_43_classes)
print()

# Splitting prepared datasets
# It will also create tags for the images: train, validation and test
fous.random_split(dataset_ts_empty, {"train": 1})
fous.random_split(dataset_ts_4_classes, {"train": 0.7, "validation": 0.2, "test": 0.1})
fous.random_split(dataset_ts_43_classes, {"train": 0.7, "validation": 0.2, "test": 0.1})


# Check point
# Showing number of samples in sub-datasets
print(dataset_ts_empty.count_sample_tags())      # {'train': 159}
print(dataset_ts_4_classes.count_sample_tags())  # {'train': 519, 'validation': 148, 'test': 74}
print(dataset_ts_4_classes.count_sample_tags())  # {'train': 519, 'validation': 148, 'test': 74}
print()

# Merging the samples together into the one dataset
_ = dataset_ts_4_classes.merge_samples(dataset_ts_empty)
_ = dataset_ts_43_classes.merge_samples(dataset_ts_empty)


# Check points
print("The datasets are successfully merged \U0001F44C")
print()
print(dataset_ts_4_classes.count_sample_tags())  # {'train': 678, 'validation': 148, 'test': 74}
print(dataset_ts_4_classes.count_sample_tags())  # {'train': 678, 'validation': 148, 'test': 74}
print()

# Check point
# Showing datasets information
print(dataset_ts_4_classes)
print("---")
print(dataset_ts_43_classes)
print()

# Loading again dataset into FiftyOne
dataset_ts_4_classes = fo.load_dataset("ts-4-classes-yolov5")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
session = fo.launch_app(dataset_ts_4_classes)


# Option 2
# Updating the session
# session.dataset = dataset_ts_4_classes

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()

# Loading again dataset into FiftyOne
dataset_ts_43_classes = fo.load_dataset("ts-43-classes-yolov5")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_ts_43_classes)


# Option 2
# Updating the session
session.dataset = dataset_ts_43_classes

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
dataset_ts_4_classes = fo.load_dataset("ts-4-classes-yolov5")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"
# label_field = "ground_truth"


# The classes to be exported
# All splits must use the same classes list
classes = ['0', '1', '2', '3']


# The splits to export
splits = ["train", "validation", "test"]


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.YOLOv5Dataset


# Export dataset in YOLO format for v5 version
for split in splits:    
    
    # Getting image samples for current split from the dataset by appropriate tag
    split_view = dataset_ts_4_classes.match_tags(split)
    
    
    # Exporting current split
    split_view.export(
                      export_dir=directory_ts_yolo_4_classes,
                      dataset_type=dataset_type,
                      label_field=label_field,
                      split=split,
                      classes=classes
                     )

print()

# Loading again dataset into FiftyOne
dataset_ts_43_classes = fo.load_dataset("ts-43-classes-yolov5")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"
# label_field = "ground_truth"


# The classes to be exported
# All splits must use the same classes list
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42']


# The splits to export
splits = ["train", "validation", "test"]


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.YOLOv5Dataset


# Export dataset in YOLO format for v5 version
for split in splits:    
    
    # Getting image samples for current split from the dataset by appropriate tag
    split_view = dataset_ts_43_classes.match_tags(split)
    
    
    # Exporting current split
    split_view.export(
                      export_dir=directory_ts_yolo_43_classes,
                      dataset_type=dataset_type,
                      label_field=label_field,
                      split=split,
                      classes=classes,
                     )

print()

#  1. Create additional directories inside 'ts_yolo' to keep TS dataset in YOLO for v4 version:
#     one with name            --> 'yolov4dataset'
#     and inside it one more       --> 'ts4classes'
#     and inside it three more         --> 'train', 'validation', 'test'
# 
#  2. Prepare paths variables to these directories:
directory_ts_yolo_4_classes_train = \
            os.path.join(os.getcwd(), "ts_yolo", "yolov4dataset", "ts4classes", "train")

directory_ts_yolo_4_classes_validation = \
            os.path.join(os.getcwd(), "ts_yolo", "yolov4dataset", "ts4classes", "validation")

directory_ts_yolo_4_classes_test = \
            os.path.join(os.getcwd(), "ts_yolo", "yolov4dataset", "ts4classes", "test")


# Loading again dataset into FiftyOne
dataset_ts_4_classes = fo.load_dataset("ts-4-classes-yolov5")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"
# label_field = "ground_truth"


# The classes to be exported
# All splits must use the same classes list
classes = ['0', '1', '2', '3']


# The splits to export
splits = ["train", "validation", "test"]


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.YOLOv4Dataset


# Export dataset in YOLO format for v4 version
for split in splits:    
    
    # Getting image samples for current split from the dataset by appropriate tag
    split_view = dataset_ts_4_classes.match_tags(split)
    
    
    # Checking if it is 'train' split
    if split == "train":
        # Exporting current split
        split_view.export(
                          export_dir=directory_ts_yolo_4_classes_train,
                          dataset_type=dataset_type,
                          label_field=label_field,
                          classes=classes,
                         )
        
        
    # Checking if it is 'validation' split
    if split == "validation":
        # Exporting current split
        split_view.export(
                          export_dir=directory_ts_yolo_4_classes_validation,
                          dataset_type=dataset_type,
                          label_field=label_field,
                          classes=classes,
                         )
    
    
    # Checking if it is 'test' split
    if split == "test":
        # Exporting current split
        split_view.export(
                          export_dir=directory_ts_yolo_4_classes_test,
                          dataset_type=dataset_type,
                          label_field=label_field,
                          classes=classes,
                         )
        
print()

#  1. Create additional directories inside 'ts_yolo' to keep TS dataset in YOLO for v4 version:
#     one with name            --> 'yolov4dataset'
#     and inside it one more       --> 'ts43classes'
#     and inside it three more         --> 'train', 'validation', 'test'
# 
#  2. Prepare paths variables to these directories:
directory_ts_yolo_43_classes_train = \
            os.path.join(os.getcwd(), "ts_yolo", "yolov4dataset", "ts43classes", "train")

directory_ts_yolo_43_classes_validation = \
            os.path.join(os.getcwd(), "ts_yolo", "yolov4dataset", "ts43classes", "validation")

directory_ts_yolo_43_classes_test = \
            os.path.join(os.getcwd(), "ts_yolo", "yolov4dataset", "ts43classes", "test")


# Loading again dataset into FiftyOne
dataset_ts_43_classes = fo.load_dataset("ts-43-classes-yolov5")


# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "detections"
# label_field = "ground_truth"


# The classes to be exported
# All splits must use the same classes list
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42']


# The splits to export
splits = ["train", "validation", "test"]


# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.YOLOv4Dataset


# Export dataset in YOLO format for v4 version
for split in splits:    
    
    # Getting image samples for current split from the dataset by appropriate tag
    split_view = dataset_ts_43_classes.match_tags(split)
    
    
    # Checking if it is 'train' split
    if split == "train":
        # Exporting current split
        split_view.export(
                          export_dir=directory_ts_yolo_43_classes_train,
                          dataset_type=dataset_type,
                          label_field=label_field,
                          classes=classes,
                         )
        
        
    # Checking if it is 'validation' split
    if split == "validation":
        # Exporting current split
        split_view.export(
                          export_dir=directory_ts_yolo_43_classes_validation,
                          dataset_type=dataset_type,
                          label_field=label_field,
                          classes=classes,
                         )
    
    
    # Checking if it is 'test' split
    if split == "test":
        # Exporting current split
        split_view.export(
                          export_dir=directory_ts_yolo_43_classes_test,
                          dataset_type=dataset_type,
                          label_field=label_field,
                          classes=classes,
                         )
        
print()

# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

# Deleting datasets from FiftyOne
fo.load_dataset("ts-4-classes-yolov5-copy").delete()
fo.load_dataset("ts-43-classes-yolov5-copy").delete()

# A name for the dataset
name = "ts-4-classes-yolov5-copy"


# The splits to export
splits = ["train", "validation", "test"]


# The type of the dataset being imported
dataset_type = fo.types.YOLOv5Dataset


# Load the dataset
dataset_ts_4_classes_copy = fo.Dataset(name)


# Using tags to mark the samples in each split
for split in splits:
    dataset_ts_4_classes_copy.add_dir(
                                      dataset_dir=directory_ts_yolo_4_classes,
                                      dataset_type=dataset_type,
                                      split=split,
                                      tags=split
                                     )

print()

# Make the dataset persistent
dataset_ts_4_classes_copy.persistent = True


# Check point
# Showing dataset information
print(dataset_ts_4_classes_copy)
print()

# Loading again dataset into FiftyOne
dataset_ts_4_classes_copy = fo.load_dataset("ts-4-classes-yolov5-copy")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_ts_4_classes_copy)


# Option 2
# Updating the session
session.dataset = dataset_ts_4_classes_copy

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()

# A name for the dataset
name = "ts-43-classes-yolov5-copy"


# The splits to export
splits = ["train", "validation", "test"]


# The type of the dataset being imported
dataset_type = fo.types.YOLOv5Dataset


# Load the dataset
dataset_ts_43_classes_copy = fo.Dataset(name)


# Using tags to mark the samples in each split
for split in splits:
    dataset_ts_43_classes_copy.add_dir(
                                       dataset_dir=directory_ts_yolo_43_classes,
                                       dataset_type=dataset_type,
                                       split=split,
                                       tags=split
                                      )

print()

# Make the dataset persistent
dataset_ts_43_classes_copy.persistent = True


# Check point
# Showing dataset information
print(dataset_ts_43_classes_copy)
print()

# Loading again dataset into FiftyOne
dataset_ts_43_classes_copy = fo.load_dataset("ts-43-classes-yolov5-copy")


# Option 1
# Launching the App 1st time, and visualizing the dataset
# Creating session instance
# session = fo.launch_app(dataset_ts_43_classes_copy)


# Option 2
# Updating the session
session.dataset = dataset_ts_43_classes_copy

# Clear the session
session.clear_dataset()


# Check point
print("The session is successfully cleared \U0001F44C")
print()

# Check point
# Showing list of loaded datasets into FiftyOne
print(fo.list_datasets())
print()

