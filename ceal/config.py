##### constants ######


# PATH definition
initial_weights_path = "./models/initial_weights.hdf5"
final_weights_path = "./models/output_weights_batch_norm.hdf5"

param_path = "./"

# Data definition
# image dimension (array)
img_rows = 512
img_cols = 512


nb_total = 69 # total number of img
nb_train = 53 # number of train img
nb_labeled = 53 # number of labeled image
nb_unlabeled = nb_train - nb_labeled # number of unlabeled images

# CEAL parameters
apply_edt = False
nb_iterations = 1 # number of active iteration

nb_step_predictions = 10

# For sample selection to present to the oracle
nb_no_detections = 1 # number of samples in the histogram area with accuracy = 0 (no detection)
nb_random = 1 # number of randomly selected samples
nb_most_uncertain = 1 # number of samples selected in the most uncertain area (highest uncertainty)
most_uncertain_rate = 1 # threshold to define the most uncertain area. 

# this 2+2+2 samples are presented to oracle for annotation

pseudo_epoch = 5
nb_pseudo_initial = 1
pseudo_rate = 2

initial_train = True
nb_initial_epochs = 150
nb_active_epochs = 2
batch_size = 10
steps_per_epoch = nb_labeled / batch_size

apply_augmentation = True
featurewise_center = False
featurewise_std_normalization= False
rotation_range= 0
horizontal_flip= False
vertical_flip =False
zca_whitening = False
rescale = 0
zoom_range= 0
channel_shift_range = 0
width_shift_range = 0
height_shift_range = 0


learning_rate = 1e-3
decay_rate = learning_rate / nb_initial_epochs
