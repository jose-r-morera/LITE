import os

# GPU performance: must be set BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                      # Disable TF warnings
os.environ.setdefault('TF_CUDNN_USE_AUTOTUNE', '1')            # cuDNN kernel auto-tuning
os.environ.setdefault('TF_GPU_THREAD_MODE', 'gpu_private')     # dedicated GPU thread pool
os.environ.setdefault('TF_GPU_THREAD_COUNT', '2')              # threads for the GPU pool

from src.utils import load_data, create_directory
import sys

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress expected retracing warnings (multiple model instances)

import pandas as pd
import numpy as np
import json
import argparse
from distutils.util import strtobool
from datetime import datetime

from src.classifiers import LITE, LITEMV

from sklearn.metrics import accuracy_score
from aeon.datasets.tsc_datasets import univariate


# Representative subset of UCR for fast benchmarking (~30 datasets)
# Stratified by: train size (tiny/small/medium/large), TS length, and number of classes
FAST_SUBSET = [
    # Tiny (≤40 train samples)
    "Coffee", "Beef", "GunPoint", "ECG200", "Wine", "ToeSegmentation1",
    # Small (41–150)
    "ArrowHead", "CBF", "BME", "FreezerRegularTrain", "Trace", "Plane",
    # Medium (151–500)
    "FacesUCR", "Fish", "Yoga", "SwedishLeaf", "ECG5000",
    "ChlorineConcentration", "Adiac", "FiftyWords",
    # Large (500+)
    "FaceAll", "StarLightCurves", "Wafer", # "FordA",
    "ElectricDevices", "Crop", "NonInvasiveFetalECGThorax1",
    # Extra coverage: long TS, many classes
    "SmoothSubspace", "ShapesAll", # "HandOutlines",
    # Multi-variation movement/gesture families
    "CricketX", "UWaveGestureLibraryX",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Choose to apply which classifier on which dataset with number of runs."
    )

    parser.add_argument(
        "--dataset",
        help="which dataset to run the experiment on.",
        type=str,
        default="Coffee",
    )

    parser.add_argument(
        "--classifier",
        help="which classifier to use",
        type=str,
        choices=["LITE","LITEMV"],
        default="LITE",
    )

    parser.add_argument("--runs", help="number of runs to do", type=int, default=5)

    parser.add_argument(
        "--output-directory",
        help="output directory parent",
        type=str,
        default="results/",
    )

    parser.add_argument(
        "--track-emissions", type=lambda x: bool(strtobool(x)), default=True
    )

    parser.add_argument(
        "--verbose", help="when true shows logs and validation tracking on each epoch", type=lambda x: bool(strtobool(x)), default=False
    )

    parser.add_argument(
        "--params", help="show the number of parameters of the classifier and exit", action="store_true"
    )

    parser.add_argument(
        "--differentiate", help="whether to add a second channel with differenced data", type=lambda x: bool(strtobool(x)), default=False
    )

    # parser.add_argument(
    #     "--lr", help="base learning rate for the Adam optimizer", type=float, default=0.001
    # )

    # parser.add_argument(
    #     "--custom-lr", help="learning rate for the hybrid (custom filter) layers. Defaults to lr/100.", type=float, default=None
    # )


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    output_directory_parent = args.output_directory
    create_directory(output_directory_parent)
    output_directory_parent = output_directory_parent + args.classifier + "/"
    create_directory(output_directory_parent)

    # Determine which datasets to run (Single vs All vs Fast)
    if args.dataset.lower() == "all":
        datasets_to_run = univariate
        print(f"Total datasets: {len(datasets_to_run)}")
    elif args.dataset.lower() == "fast":
        datasets_to_run = FAST_SUBSET
        print(f"Fast benchmark: {len(datasets_to_run)} representative datasets")
    else:
        datasets_to_run = [args.dataset]

    # Iterate through the selected datasets
    for i, dataset_name in enumerate(datasets_to_run):
        # Update args.dataset so internal logic uses the current dataset name
        args.dataset = dataset_name
        
        # --- TIMESTAMP: Start of Dataset ---
        current_time = datetime.now().strftime("%H:%M")
        
        if not args.params:
            print(f"[{current_time}] Starting execution for {args.dataset} ({i+1}/{len(datasets_to_run)})")

        xtrain, ytrain, xtest, ytest = load_data(file_name=args.dataset, differentiate=args.differentiate)
        
        length_TS = int(xtrain.shape[1])
        n_channels = int(xtrain.shape[2])
        n_classes = len(np.unique(ytrain))

        if args.params:
            if args.classifier == "LITE":
                clf = LITE(
                    output_directory="/tmp/",
                    length_TS=length_TS,
                    n_channels=n_channels,
                    n_classes=n_classes,
                    verbose=False,
                    # lr=args.lr,
                    # custom_lr=args.custom_lr,
                )
            elif args.classifier == "LITEMV":
                clf = LITEMV(
                    output_directory="/tmp/",
                    length_TS=length_TS,
                    n_channels=n_channels,
                    n_classes=n_classes,
                    verbose=False,
                    # lr=args.lr,
                    # custom_lr=args.custom_lr,
                )
            
            print(f"\n{'='*50}")
            print(f"Classifier: {args.classifier}")
            print(f"Dataset:    {args.dataset}")
            print(f"Dimensions: {length_TS} (length) x {n_channels} (channels)")
            print(f"Classes:    {n_classes}")
            print(f"{'-'*50}")
            
            total_params = clf.model.count_params()
            trainable_params = np.sum([tf.keras.backend.count_params(w) for w in clf.model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            print(f"Total Parameters:         {total_params:,}")
            print(f"Trainable Parameters:     {trainable_params:,}")
            print(f"Non-trainable Parameters: {non_trainable_params:,}")
            print(f"{'='*50}\n")
            
            if i == len(datasets_to_run) - 1:
                sys.exit(0)
            continue

        # Check if the dataset has already been run and if so, skip it.
        if os.path.exists(output_directory_parent + "results_ucr.csv"):
            df = pd.read_csv(output_directory_parent + "results_ucr.csv")

            file_names = list(df["dataset"])
            if args.dataset in file_names:
                print(f"Results for {args.dataset} already exist. Skipping.")
                continue # Use continue instead of exit() to allow loop to proceed to next dataset
        else:
            columns = [
                "dataset",
                args.classifier + "-mean",
                args.classifier + "-std",
                args.classifier + "Time",
                "Training Time (s)-mean",
                "Testing Time (s)-mean",
            ]
            if args.track_emissions:
                columns += [
                    "CO2eq Emissions (kg)-mean",
                    "Energy Consumption (kWh)-mean",
                    "Country",
                    "Region",
                ]
            df = pd.DataFrame(columns=columns)

        ypred = np.zeros(shape=(len(ytest), len(np.unique(ytest))))

        Scores = []
        training_time = []
        testing_time = []

        if args.track_emissions:
            co2_consumption = []
            energy_consumption = []

        for _run in range(args.runs):
            output_directory = output_directory_parent + "run_" + str(_run) + "/"
            create_directory(output_directory)
            output_directory = output_directory + args.dataset + "/"
            create_directory(output_directory)

            if args.classifier == "LITE":
                clf = LITE(
                    output_directory=output_directory,
                    length_TS=length_TS,
                    n_channels=n_channels,
                    n_classes=len(np.unique(ytrain)),
                    verbose=args.verbose,
                    # lr=args.lr,
                    # custom_lr=args.custom_lr,
                )
            elif args.classifier == "LITEMV":
                clf = LITEMV(
                    output_directory=output_directory,
                    length_TS=length_TS,
                    n_channels=n_channels,
                    n_classes=len(np.unique(ytrain)),
                    verbose=args.verbose,
                    # lr=args.lr,
                    # custom_lr=args.custom_lr,
                )
            else:
                raise ValueError("Choose an existing classifier.")

            if not os.path.exists(output_directory + "loss.pdf"):
                if args.track_emissions:
                    dict_emissions = clf.fit_and_track_emissions(
                        xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest, plot_test=args.verbose
                    )
                else:
                    clf.fit(
                        xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest, plot_test=args.verbose
                    )

            else:
                if args.track_emissions:
                    with open(output_directory + "dict_emissions.json") as json_file:
                        dict_emissions = json.load(json_file)

            training_time.append(clf.train_duration)

            if args.track_emissions:
                co2_consumption.append(dict_emissions["co2"])
                energy_consumption.append(dict_emissions["energy"])

            y_pred, acc, duration_test = clf.predict(xtest=xtest, ytest=ytest)
            testing_time.append(duration_test)

            # Accumulate predictions for ensemble (later we get the average)
            ypred = ypred + y_pred

            Scores.append(acc)

        # Get the average for the ensemble (LITE-time)
        ypred = ypred / (args.runs * 1.0)
        ypred = np.argmax(ypred, axis=1)

        acc_Time = accuracy_score(y_true=ytest, y_pred=ypred, normalize=True)

        row = {
            "dataset": args.dataset,
            args.classifier + "-mean": np.mean(Scores),
            args.classifier + "-std": np.std(Scores),
            args.classifier + "Time": acc_Time,
            "Training Time (s)-mean": np.mean(training_time),
            "Testing Time (s)-mean": np.mean(testing_time),
        }
        if args.track_emissions:
            row.update({
                "CO2eq Emissions (kg)-mean": np.mean(co2_consumption),
                "Energy Consumption (kWh)-mean": np.mean(energy_consumption),
                "Country": str(dict_emissions["country_name"]),
                "Region": str(dict_emissions["region"]),
            })
        df.loc[len(df)] = row

        # Save the mean results across all n runs in a csv file.
        df.to_csv(output_directory_parent + "results_ucr.csv", index=False)

    # --- TIMESTAMP: End of All ---
    final_time = datetime.now().strftime("%H:%M")
    print(f"[{final_time}] All datasets finished.")