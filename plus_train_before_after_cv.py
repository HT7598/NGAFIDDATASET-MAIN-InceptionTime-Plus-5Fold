import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ngafiddataset.dataset.dataset import NGAFID_Dataset_Manager
from ngafiddataset.dataset.utils import (
    get_dict_mod,
    get_scaler,
    get_slice,
    replace_nan_w_zero,
    to_dict_of_list,
)


tfk = tf.keras
tfkl = tf.keras.layers


class Classifier_INCEPTION:
    def __init__(
        self,
        input_shape,
        nb_classes,
        build=True,
        batch_size=64,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        nb_epochs=1500,
        two_output=False,
        mode=None,
    ):
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.two_output = two_output
        self.mode = mode

        if build:
            self.model = self.build_model(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation="linear"):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_size_s:
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=self.nb_filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding="same")(input_tensor)
        conv_6 = tf.keras.layers.Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool_1)

        conv_list.append(conv_6)
        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = tf.keras.layers.Input(input_shape, name="data")
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = []
        outputs.append(tf.keras.layers.Dense(nb_classes, activation="softmax", name="target_class")(gap_layer))

        if self.two_output:
            outputs.append(tf.keras.layers.Dense(1, activation="sigmoid", name="before_after")(gap_layer))

        if self.mode == "before_after":
            outputs = []
            outputs.append(tf.keras.layers.Dense(1, activation="sigmoid", name="before_after")(gap_layer))

        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
        return model


def build_dataset(dm, examples, batch_size, shuffle, repeat, drop_remainder):
    ds = tf.data.Dataset.from_tensor_slices(to_dict_of_list(examples))
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(shuffle, seed=42, reshuffle_each_iteration=True)

    ds = ds.map(get_dict_mod("data", get_scaler(dm.maxs, dm.mins)))
    ds = ds.map(get_dict_mod("data", replace_nan_w_zero))
    ds = ds.map(get_dict_mod("data", lambda x: tf.cast(x, tf.float32)))
    ds = ds.map(lambda x: (x["data"], x["before_after"]))
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_fold(model, val_ds, y_true):
    y_prob = model.predict(val_ds, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    y_true = np.asarray(y_true).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics, y_prob, y_pred


def build_model(model_name, input_shape, mode):
    model_name = model_name.lower()

    if model_name in {"inception", "itime", "inceptiontime","inception_plus"}:
        return Classifier_INCEPTION(
            input_shape=input_shape,
            nb_classes=1,
            two_output=False,
            mode=mode,
        ).model

    raise ValueError(f"Unsupported model name: {model_name}")


def run_fold(dm, fold, args, output_dir):
    train_examples = get_slice(dm.data_dict, fold=fold, reverse=True)
    val_examples = get_slice(dm.data_dict, fold=fold, reverse=False)

    train_ds = build_dataset(
        dm,
        train_examples,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        repeat=True,
        drop_remainder=False,
    )
    val_ds = build_dataset(
        dm,
        val_examples,
        batch_size=args.batch_size,
        shuffle=0,
        repeat=False,
        drop_remainder=False,
    )

    eval_batch_size = args.eval_batch_size or args.batch_size
    eval_ds = build_dataset(
        dm,
        val_examples,
        batch_size=eval_batch_size,
        shuffle=0,
        repeat=False,
        drop_remainder=False,
    )

    steps_per_epoch = args.steps_per_epoch or math.ceil(len(train_examples) / args.batch_size)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed + fold)

    model = build_model(
        model_name=args.model_name,
        input_shape=(4096, 23),
        mode="before_after",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_acc")],
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    )

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    weights_path = fold_dir / "best_before_after.weights.h5"
    history_path = fold_dir / "history.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history_df.index
    history_df.to_csv(history_path, index=False)

    if weights_path.exists():
        model.load_weights(weights_path)

    y_true = [example["before_after"] for example in val_examples]
    metrics, y_prob, y_pred = evaluate_fold(model, eval_ds, y_true)

    pred_df = pd.DataFrame(
        {
            "id": [example["id"] for example in val_examples],
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)

    metrics["fold"] = fold
    metrics["n_train"] = len(train_examples)
    metrics["n_val"] = len(val_examples)

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="NGAFID before/after 5-fold cross-validation trainer")
    parser.add_argument("--dataset-name", default="2days")
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--model-name", default="inception")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--shuffle", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--only-fold", type=int, default=None, choices=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(args.seed)

    output_dir = Path(args.output_root) / args.model_name / "before_after_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = NGAFID_Dataset_Manager(args.dataset_name, destination=args.dataset_dir)
    dm.data_dict = dm.construct_data_dictionary()

    folds = [args.only_fold] if args.only_fold is not None else list(range(5))
    results = []

    for fold in folds:
        print(f"\n===== Running fold {fold} =====")
        metrics = run_fold(dm, fold, args, output_dir)
        print(
            "Fold {fold}: accuracy={accuracy:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}".format(
                **metrics
            )
        )
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("fold")
    results_df.to_csv(output_dir / "cv_results.csv", index=False)

    summary = {
        "accuracy_mean": float(results_df["accuracy"].mean()),
        "accuracy_std": float(results_df["accuracy"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "f1_mean": float(results_df["f1"].mean()),
        "f1_std": float(results_df["f1"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "roc_auc_mean": float(results_df["roc_auc"].mean()),
        "roc_auc_std": float(results_df["roc_auc"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "folds": folds,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Cross-validation summary =====")
    print(f"model_name = {args.model_name}")
    print(results_df[["fold", "accuracy", "f1", "roc_auc"]])
    print(
        "accuracy = {accuracy_mean:.4f} ± {accuracy_std:.4f}\n"
        "f1       = {f1_mean:.4f} ± {f1_std:.4f}\n"
        "roc_auc  = {roc_auc_mean:.4f} ± {roc_auc_std:.4f}".format(**summary)
    )


if __name__ == "__main__":
    main()
