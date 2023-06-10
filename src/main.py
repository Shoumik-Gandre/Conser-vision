from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import yaml
from torch.backends import cudnn

from src import enumerations, l2sp
from src.baseline import train_baseline
from src.baseline.hyperparams import BaseHyperparams
from src.predict import predict_animal
from src.enumerations import TransferTechnique


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args: Namespace) -> None:
    print(args)
    set_seeds(42)
    match args.mode:
        case 'train':
            match args.transfer:
                case TransferTechnique.BASE:
                    with open(args.hyperparams, 'r') as file:
                        hyperparams_data = yaml.safe_load(file)
                    hyperparams = BaseHyperparams(**hyperparams_data)
                    if hasattr(args, 'batch_size'):
                        hyperparams.batch_size = args.batch_size
                    if hasattr(args, 'num_epochs'):
                        hyperparams.num_epochs = args.num_epochs

                    train_baseline(
                        features_csv=args.features_csv_path,
                        labels_csv=args.labels_csv_path,
                        images_dir=args.image_dir,
                        model_path=args.model_path,
                        model_arch=args.architecture,
                        hyperparams=hyperparams
                    )
                case TransferTechnique.FREEZE:
                    print('train freeze')

                case TransferTechnique.L2_SP:
                    with open(args.hyperparams, 'r') as file:
                        hyperparams_data = yaml.safe_load(file)
                    hyperparams = l2sp.L2SPHyperparams(**hyperparams_data)

                    if hasattr(args, 'batch_size'):
                        hyperparams.batch_size = args.batch_size
                    if hasattr(args, 'num_epochs'):
                        hyperparams.num_epochs = args.num_epochs

                    print(hyperparams)

                    l2sp.l2sp_train(
                        features_csv=args.features_csv_path,
                        labels_csv=args.labels_csv_path,
                        images_dir=args.image_dir,
                        model_path=args.model_path,
                        model_arch=args.architecture,
                        hyperparams=hyperparams
                    )
                case TransferTechnique.BSS:
                    print('train bss')

                case TransferTechnique.DELTA:
                    print('train delta')

                case TransferTechnique.CO_TUNING:
                    print('train co-tuning')

        case 'predict':

            match args.transfer:
                case TransferTechnique.BASE | TransferTechnique.L2_SP:
                    predict_animal(
                        model_path=args.model_path,
                        features_csv=args.features_csv_path,
                        images_dir=args.image_dir,
                        prediction_path=args.predictions,
                        architecture=args.architecture,
                        batch_size=args.batch_size,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )


def handle_args() -> Namespace:
    main_parser = ArgumentParser()
    subparser = main_parser.add_subparsers(title='mode', dest='mode', required=True)
    # Train mode
    train_parser = subparser.add_parser('train', help='train mode')
    train_parser.add_argument('--image-dir', '-d',
                              dest='image_dir',
                              required=True,
                              help='path to dir containing train_features dir')
    train_parser.add_argument('--features', '-x',
                              dest='features_csv_path',
                              required=True,
                              help='path to train features csv')
    train_parser.add_argument('--labels', '-y',
                              dest='labels_csv_path',
                              required=True,
                              help='path to train labels csv')
    train_parser.add_argument('--model-path', '-o',
                              required=True,
                              help='model save path')
    train_parser.add_argument('--arch', '-a',
                              choices=list(enumerations.Architectures),
                              type=enumerations.Architectures,
                              dest='architecture',
                              required=True,
                              help='model architecture in use')
    train_parser.add_argument('--transfer', '-t',
                              choices=list(enumerations.TransferTechnique),
                              type=enumerations.TransferTechnique,
                              dest='transfer',
                              required=True,
                              help='transfer learning strategy')
    train_parser.add_argument('--num-epochs',
                              type=int,
                              dest='num_epochs',
                              )
    train_parser.add_argument('--batch-size',
                              type=int,
                              dest='batch_size',
                              )
    train_parser.add_argument('--checkpoint',
                              action='store_true',
                              dest='checkpoint',
                              help='this is a flag, if this is enabled, then we load the model path first as checkpoint'
                              )
    train_parser.add_argument('--hyperparams',
                              dest='hyperparams',
                              required=True
                              )

    # Predict mode
    predict_parser = subparser.add_parser('predict', help='predict mode')
    predict_parser.add_argument('--image-dir', '-d',
                                dest='image_dir',
                                required=True,
                                help='path to dir containing test_features dir')
    predict_parser.add_argument('--features', '-x',
                                dest='features_csv_path',
                                required=True,
                                help='path to test features csv')
    predict_parser.add_argument('--model-path', '-i',
                                required=True,
                                help='model load path')
    predict_parser.add_argument('--predict', '-o',
                                dest='predictions',
                                required=True,
                                help='prediction file save path')
    predict_parser.add_argument('--arch', '-a',
                                choices=list(enumerations.Architectures),
                                type=enumerations.Architectures,
                                dest='architecture',
                                required=True,
                                help='model architecture in use')
    predict_parser.add_argument('--transfer', '-t',
                                choices=list(enumerations.TransferTechnique),
                                type=enumerations.TransferTechnique,
                                dest='transfer',
                                required=True,
                                help='transfer learning strategy')
    predict_parser.add_argument('--batch-size',
                                type=int,
                                dest='batch_size',
                                required=True
                                )

    return main_parser.parse_args()


if __name__ == '__main__':
    main(handle_args())
