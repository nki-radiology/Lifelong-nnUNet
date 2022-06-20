#########################################################################################################
#----------This class represents the Evaluation of networks using the extended nnUNet trainer-----------#
#----------                                     version.                                     -----------#
#########################################################################################################

import os, argparse, nnunet_ext
from nnunet_ext.evaluation.evaluator import Evaluator
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.paths import evaluation_output_dir, default_plans_identifier
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

EXT_MAP = dict()
# -- Extract all extensional trainers in a more generic way -- #
extension_keys = [x for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training")) if 'py' not in x]
for ext in extension_keys:
    trainer_name = [x[:-3] for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training", ext)) if '.py' in x][0]
    # trainer_keys.extend(trainer_name)
    EXT_MAP[trainer_name] = ext
# -- Add standard trainers as well -- #
EXT_MAP['nnViTUNetTrainer'] = None
EXT_MAP['nnUNetTrainerV2'] = 'standard'
EXT_MAP['nnViTUNetTrainerCascadeFullRes'] = None


def run_evaluation():
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert evaluation_output_dir is not None, "Before running any evaluation, please specify the Evaluation folder (EVALUATION_FOLDER) as described in the paths.md."

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a multi head, sequential, rehearsal, ewc or lwf
    
    # -- nnUNet arguments untouched --> Should not intervene with sequential code, everything should work -- #
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    
    # -- Additional arguments specific for multi head training -- #
    parser.add_argument("-f", "--folds",  action='store', type=str, nargs="+",
                        help="Specify on which folds to train on. Use a fold between 0, 1, ..., 4 or \'all\'", required=True)
    parser.add_argument("-trained_on", action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network has trained with to specify the correct path to the networks. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-use_model", "--use", action='store', type=str, nargs="+",
                        help="Specify a list of task ids that specify the exact network that should be used for evaluation. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument('-use_head', action='store', type=str, nargs=1, required=False, default=None,
                        help='Specify which head to use for the evaluation of tasks the network is not trained on. When using a non nn-UNet extension, that' +
                              'is not necessary. If this is not set, always the latest trained head will be used.')
    parser.add_argument("--fp32_used", required=False, default=False, action="store_true",
                        help="Specify if mixed precision has been used during training or not")
    parser.add_argument("-evaluate_on",  action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network will be evaluated on. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('--store_csv', required=False, default=False, action="store_true",
                        help='Set this flag if the validation data and any other data if applicable should be stored'
                            ' as a .csv file as well. Default: .csv are not created.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1], choices=[1, 2, 3, 4],
                        help='Select the ViT input building version. Currently there are only four'+
                            ' possibilities: 1, 2, 3 or 4.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base', choices=['base', 'large', 'huge'],
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. '+
                             'Note that then the flags -v, -v_type and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will have task specific Layer Norms.')
    parser.add_argument('--no_transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head should not be initialized using the last head'
                            ' during training, ie. the very first head from the initialization of the class is used.'
                            ' Default: The previously trained head is used as initialization of the new head.')
    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. '+
                             'When this is set, more than one GPU needs to be provided when using -d.')
    parser.add_argument('--always_use_last_head', action='store_true', default=False,
                        help='If this is set, during the evaluation, always the last head will be used, '+
                             'for every dataset the evaluation is performed on. When an extension network was trained with '+
                             'the -transfer_heads flag then this should be set, i.e. nnUNetTrainerSequential or nnUNetTrainerFrozendViT.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')
    parser.add_argument('--FeatScale', action='store_true', default=False,
                        help='Set this flag if Feature Scale should be used for the ViT.')
    parser.add_argument('--AttnScale', action='store_true', default=False,
                        help='Set this flag if Attention Scale should be used for the ViT.')
    parser.add_argument('--FFT', action='store_true', default=False,
                        help='Set this flag if MSA should be replaced with FFT Blocks (every 2nd layer only).')
    parser.add_argument("-FFT_filter", action='store', type=str, nargs=1, default=None, choices=['high_basic', 'high_advanced'],
                        help='Specify if and which FFT filtering will be performed (has nothing to do with --FFT). high_basic applies a simple high pass filter,'
                            +'whereas high_advanced is a more sophisticated version.')
    parser.add_argument('-filter_every', action='store', type=int, nargs=1, required=False, default=10,
                        help='Specify the interval after which batch iteration FFT filtering will be applied.'
                            ' Default: If FFT_filter is set, the filtering will be done every 10th batch iteration.')
    parser.add_argument('-filter_rate', action='store', type=float, nargs=1, required=False, default=0.31,
                        help='Specify the amount of skipped FFT information during the high_basic filtering.'
                            ' Default: If FFT_filter is set to high_basic, 31 percent of low-pass information is skipped.')
    parser.add_argument('-f_map_type', action='store', type=str, nargs=1, required=False, default='none', choices=['none', 'basic', 'gauss_1', 'gauss_10', 'gauss_100'],
                        help='Specify if fourrier feature mapping should be used before the ViTs MLP module along with the type.'
                            ' Note that the argument none makes literally no modification. Default: No mapping will be performed.')
    parser.add_argument('-replace_every', action='store', type=int, nargs=1, required=False, default=None,
                        help='Specify after which amount of MSA a Convolutional smoothing should be used instead.')
    parser.add_argument('-do_n_blocks', action='store', type=int, nargs=1, required=False, default=None,
                        help='Specify the amount of Convolutional smoothing blocks.')
    parser.add_argument('-smooth_temp', action='store', type=float, nargs=1, required=False, default=10,
                        help='Specify the smoothing temperature for Convolutional smoothing blocks. Default: 10.')
    parser.add_argument('--special', action='store_true', default=False,
                        help='Set this flag if our special FFT ViT Unet with multiple heads and cross-attention should be used.')
    parser.add_argument('--cbam', action='store_true', default=False,
                        help='Set this flag to alternately replace a MSA block with a CBAM block.')
    parser.add_argument('--adaptive', required=False, default=False, action="store_true",
                        help='Set this flag if the EWC loss has been changed during the frozen training process (ewc_lambda*e^{-1/3}). '
                             ' Default: The EWC loss will not be altered. --> Makes only sense with our nnUNetTrainerFrozEWC trainer.')
    parser.add_argument('--include_training_data', action='store_true', default=False,
                        help='Set this flag if the evaluation should also be done on the training data.')


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    always_use_last_head = args.always_use_last_head
    
    # -- Extract the arguments specific for all trainers from argument parser -- #
    trained_on = args.trained_on    # List of the tasks that helps to navigate to the correct folder, eg. A B C
    use_model = args.use            # List of the tasks representing the network to use, e. use A B from folder A B C
    evaluate_on = args.evaluate_on  # List of the tasks that should be used to evaluate the model
    use_head = args.use_head        # One task specifying which head should be used
    if isinstance(use_head, list):
        use_head = use_head[0]
        
    # -- Extract further arguments -- #
    save_csv = args.store_csv
    fold = args.folds
    cuda = args.device
    mixed_precision = not args.fp32_used
    transfer_heads = not args.no_transfer_heads
    adaptive = args.adaptive

    # -- Extract ViT specific flags to as well -- #
    use_vit = args.use_vit
    ViT_task_specific_ln = args.task_specific_ln
    
    # -- Extract the vit_type structure and check it is one from the existing ones -- #s
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    # assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'

    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT

    # -- Scaling flags -- #
    FeatScale = args.FeatScale
    AttnScale = args.AttnScale
    useFFT = args.FFT
    f_map_type = args.f_map_type[0] if isinstance(args.f_map_type, list) else args.f_map_type

    conv_smooth = [args.replace_every[0] if isinstance(args.replace_every, list) else args.replace_every,
                   args.do_n_blocks[0] if isinstance(args.do_n_blocks, list) else args.do_n_blocks,
                   args.smooth_temp[0] if isinstance(args.smooth_temp, list) else args.smooth_temp]
    conv_smooth = None if conv_smooth[0] is None or conv_smooth[1] is None else conv_smooth
    special = args.special
    cbam = args.cbam
    
    # -- Filtering specific arguments -- #
    FFT_filter = args.FFT_filter[0] if isinstance(args.FFT_filter, list) else args.FFT_filter
    filter_every = args.filter_every[0] if isinstance(args.filter_every, list) else args.filter_every
    filter_rate = args.filter_rate[0] if isinstance(args.filter_rate, list) else args.filter_rate
    
    # -- Assert if device value is ot of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error

    # -- Check if the user wants to split the network onto multiple GPUs -- #
    split_gpu = args.use_mult_gpus
    if split_gpu:
        assert len(cuda) > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'
        
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Set bool if user wants to use train data during eval as well -- #
    use_all_data = args.include_training_data

    # -- Extract the desired version, only considered in case of ViT Trainer -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    
    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(5))
    else: # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int, list)), "To Evaluate multiple tasks with {} trainer, only one or multiple folds specified as integers are allowed..".format(network_trainer)

    # -- Build all necessary task lists -- #
    tasks_for_folder = list()
    use_model_w_tasks = list()
    evaluate_on_tasks = list()
    if use_head is not None:
        use_head = convert_id_to_task_name(int(use_head)) if not use_head.startswith("Task") else use_head
    for idx, t in enumerate(trained_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        tasks_for_folder.append(t)
    for idx, t in enumerate(use_model):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        use_model_w_tasks.append(t)
    for idx, t in enumerate(evaluate_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        evaluate_on_tasks.append(t)

    char_to_join_tasks = '_'
    
    # ---------------------------------------------
    # Evaluate for each task and all provided folds
    # ---------------------------------------------
    evaluator = Evaluator(network, network_trainer, (tasks_for_folder, char_to_join_tasks), (use_model_w_tasks, char_to_join_tasks), 
                          version, vit_type, plans_identifier, mixed_precision, EXT_MAP[network_trainer], save_csv, transfer_heads,
                          use_vit, False, ViT_task_specific_ln, do_LSA, do_SPT, FeatScale, AttnScale, filter_rate, FFT_filter, filter_every,
                          useFFT, f_map_type, conv_smooth, special, cbam)
    evaluator.evaluate_on(fold, evaluate_on_tasks, use_head, always_use_last_head, adaptive=adaptive, use_all_data=use_all_data)

# -- Main function for setup execution -- #
def main():
    run_evaluation()

if __name__ == "__main__":
    run_evaluation()