from nnunet_ext.run.run import nnUNet_extract_uncertainties, \
    nnUNet_extract_features, nnUNet_extract_TTA_outputs, \
    nnUNet_extract_non_softmaxed_outputs, nnUNet_extract_outputs, \
    nnUNet_extract_MCDO_outputs, nnUNet_estimate_gaussian


def configure_dataset(inputs_path, pred_dataset_name, inputs_path_train_data, pred_dataset_name_train, feature_paths,
                      task_id, fold_ix, model_type, checkpoint, mahal_features, targets_path, train_ds_names, store_ds_names, label, nr_labels, temperatures, patch_size):
    """ Set all your dataset variables and finally extract the uncertainty estimation.

     inputs_path: this should be the dataset for which you want to extract uncertainties
     pred_dataset_name: name of the dataset for which outputs are to be extracted (name of images stored in inputs_path)
     inputs_path_train_data: Set inputs_path for the nnUNet training data (imageTr)
     feature_paths: paths to the feature names, e.g.
        ['conv_blocks_context.6.blocks.1.conv']
     param task_id: Task (dataset name) of the pre-trained model that is loaded
     fold_ix: Fold of the model instance that is restored, often 0
     model_type: Network type, e.g. 3d_fullres
     checkpoint: Checkpoint from which to restore the model state
     mahal_features: paths to the feature names, e.g.
        {'CB6': ['conv_blocks_context.6.blocks.1.conv']}
     targets_path: path name for the directory where the labels are stored for the input images ('pred_dataset_name')
     train_ds_names: the folder name where the features of the training set are stored
     store_ds_names: the folder name where the features of the new data set is stored ('pred_dataset_name')
     label: label for the class of interest
     nr_labels: number of classes in the segmentation masks (for
        one foreground and background nr_labels == 2)
     temperatures: temperatures for which temperature scaling is calculated
     patch_size: patch_size

    !!! IMPORTANT !!!
    First, regular predictions should be extracted so that the performance of the model on the new data can be assessed

    For the Mahalanobis distance, features must be extracted and the gaussian distance should be calcualted.
        First: Extract features from the 'training data' of the nnUNet model (imagesTr) and also features from the input data
        Second: The gaussian distance for the new data is calculated, with respect to the 'training data'.

    Then extract all other outputs, before the uncertainty can be calculated

    Once all outputs are extracted, the uncertainties can be extracted; which will result in a .csv file
    """

    nnUNet_extract_features(inputs_path=inputs_path_train_data, pred_dataset_name=pred_dataset_name_train, feature_paths=feature_paths, task_id=task_id, model_type=model_type, checkpoint=checkpoint)
    nnUNet_extract_features(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, feature_paths=feature_paths, task_id=task_id, model_type=model_type, checkpoint=checkpoint)
    nnUNet_estimate_gaussian(task_id=task_id, fold_ix=fold_ix, train_ds_names=train_ds_names, store_ds_names=store_ds_names, feature_paths=feature_paths)

    nnUNet_extract_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, task_id=task_id, model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix)

    nnUNet_extract_MCDO_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, mcdo_ix=0, task_id=task_id, model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix)
    nnUNet_extract_MCDO_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, mcdo_ix=1, task_id=task_id, model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix)
    nnUNet_extract_MCDO_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, mcdo_ix=2, task_id=task_id, model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix)
    nnUNet_extract_MCDO_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, mcdo_ix=3, task_id=task_id, model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix)

    nnUNet_extract_TTA_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, tta_ix=0, task_id=task_id, model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix)

    nnUNet_extract_non_softmaxed_outputs(inputs_path=inputs_path, pred_dataset_name=pred_dataset_name, task_id=task_id, model_type=model_type, checkpoint='model_best', fold_ix=fold_ix)

    nnUNet_extract_uncertainties(pred_dataset_name=pred_dataset_name, task_id=task_id, fold_ix=fold_ix,
                                 mahal_features=mahal_features, targets_path=targets_path, label=label,
                                 nr_labels=nr_labels, temperatures=temperatures, patch_size=patch_size)

#%%
# Variables to adjust according to your own dataset

inputs_path = r'/data/groups/beets-tan/j.greidanus/nnUNet_raw_data_base/nnUNet_raw_data/NABUCCO_baseline'               # Path name of the dataset for which you want to extract uncertainties
pred_dataset_name = "NABUCCO_baseline"                                                                                  # Path name of the dataset for which outputs are to be extracted (name of images stored in inputs_path)
inputs_path_train_data = r'/data/groups/beets-tan/j.greidanus/nnUNet_raw_data_base/nnUNet_raw_data/Task071_BTbase_ont/imagesTr/' # Path name for the nnUNet training data (imageTr)
pred_dataset_name_train = ['Task071_BTbase_ont']                                                                        # Path name of the train image dataset for which outputs are to be extracted (name of images stored in inputs_path_training)
feature_paths = ['conv_blocks_context.4.blocks.1.conv']                                                                 # Define paths to the feature names
task_id = "071"                                                                                                         # Define task ID of the nnUNet model that you want to use
fold_ix = "0"                                                                                                           # Specify which fold instance you want to use
model_type = '3d_fullres'                                                                                               # Define model configuration type
checkpoint = 'model_final_checkpoint'                                                                                   # Define the checkpoint from which to restore the model state
mahal_features = {'CB4': ['conv_blocks_context.4.blocks.1.conv']}                                                       # Define Paths to the feature names
targets_path = r"/data/groups/beets-tan/j.greidanus/nnUNet_raw_data_base/nnUNet_raw_data/NABUCCO_baseline/labels/"      # Define the directory where the labels are stored for the input images ('pred_dataset_name')
train_ds_names = ['Task071_BTbase_ont']                                                                                 # Folder name where the features of the training set are stored
store_ds_names = ['NABUCCO_baseline']                                                                                   # Folder name where the features of the new data set is stored ('pred_dataset_name')
label = 1                                                                                                               # Label for the class of interest
nr_labels = 2                                                                                                           # Number of classes in the segmentation masks
temperatures = [10]                                                                                                     # Define the temperatures for which temperature scaling is calculated
patch_size=[96, 160, 160]

configure_dataset(inputs_path, pred_dataset_name, inputs_path_train_data, pred_dataset_name_train, feature_paths,
                  task_id, fold_ix, model_type, checkpoint, mahal_features, targets_path, train_ds_names,
                  store_ds_names, label, nr_labels, temperatures, patch_size)