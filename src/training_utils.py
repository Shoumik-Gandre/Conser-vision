import pathlib
import typing

import pandas


def load_training_data(features_csv: str, labels_csv: str, images_dir: str,
                       ) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]:

    features = pandas.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))
    train_labels = pandas.read_csv(labels_csv, index_col="id")

    y = train_labels
    x = features.loc[y.index].filepath.to_frame()

    return x, y
