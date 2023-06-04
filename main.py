from argparse import ArgumentParser, Namespace

import torch

from baseline import train_baseline
from baseline.predict import predict_baseline


def main(args: Namespace) -> None:
    match args.mode:
        case 'train':
            assert hasattr(args, 'train_data_csv')
            assert hasattr(args, 'train_labels_csv')
            assert hasattr(args, 'train_images_dir')
            match args.solution:
                case 'baseline':
                    train_baseline(
                        features_csv=args.train_data_csv,
                        labels_csv=args.train_labels_csv,
                        images_dir=args.train_images_dir,
                        model_dir=args.model_path,
                        epochs=args.num_epochs,
                        batch_size=args.batch_size,
                    )

        case 'predict':
            assert hasattr(args, 'test_data_csv')
            assert hasattr(args, 'test_images_dir')
            assert hasattr(args, 'prediction_path')
            assert hasattr(args, 'model_path')
            match args.solution:
                case 'baseline':
                    predict_baseline(
                        model_path=args.model_path,
                        features_csv=args.test_data_csv,
                        images_dir=args.test_images_dir,
                        prediction_path=args.prediction_path,
                        batch_size=args.batch_size,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    argument_parser.add_argument('--solution', choices=['baseline'], required=True)

    argument_parser.add_argument('--train-data-csv', type=str)
    argument_parser.add_argument('--train-labels-csv', type=str)
    argument_parser.add_argument('--train-images-dir', type=str)

    argument_parser.add_argument('--test-data-csv', type=str)
    argument_parser.add_argument('--test-images-dir', type=str)

    argument_parser.add_argument('--model-path', type=str)
    argument_parser.add_argument('--prediction-path', type=str)

    argument_parser.add_argument('--batch-size', type=int, default=32)
    argument_parser.add_argument('--num-epochs', type=int, default=2)
    main(argument_parser.parse_args())
