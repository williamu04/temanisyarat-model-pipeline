import random
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from src.config import CONFIG, DATA_DIR, OUTPUT_DIR, MODEL_INPUT_DIM
from src.data import (
    scan_dataset_with_signers,
    build_tf_dataloaders,
    create_tf_dataset,
)
from src.model import build_mobile_sign_gru
from src.train import train_tf_model
from src.evaluate import (
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    save_evaluation_results,
)
from src.export import (
    convert_saved_model_to_tflite,
    export_selfcontained_tflite,
    benchmark_tf_model,
    benchmark_tflite_model,
)


def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    all_paths, all_labels, all_signer_ids = scan_dataset_with_signers(DATA_DIR)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    unique_signers = sorted(set(all_signer_ids))
    N = len(unique_signers)
    print(f"Total signers: {N} — {len(all_paths)} samples")

    rng = random.Random(42)
    unique_signers_shuffled = unique_signers.copy()
    rng.shuffle(unique_signers_shuffled)

    fold_results = []
    best_test_acc = -1.0
    best_model = None
    best_fold_idx = 0
    best_eval_results = None
    best_history_obj = None
    best_test_signer = None

    for fold, test_signer in enumerate(unique_signers_shuffled):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{N} — test: {test_signer}")
        print(f"{'=' * 60}")

        train_val_paths = [
            p for p, s in zip(all_paths, all_signer_ids) if s != test_signer
        ]
        train_val_labels = [
            l for l, s in zip(all_labels, all_signer_ids) if s != test_signer
        ]
        train_val_signer_ids = [s for s in all_signer_ids if s != test_signer]
        train_val_signers = sorted(set(train_val_signer_ids))
        k_folds = min(len(train_val_signers), 5)

        train_ds, val_ds, le, num_classes, signer_info = build_tf_dataloaders(
            DATA_DIR,
            max_len=CONFIG["max_len"],
            batch_size=CONFIG["batch_size"],
            k_folds=k_folds,
            current_fold=0,
            le_fitted=label_encoder,
            augment=True,
            paths=train_val_paths,
            labels=train_val_labels,
            signer_ids=train_val_signer_ids,
        )

        test_paths = [p for p, s in zip(all_paths, all_signer_ids) if s == test_signer]
        test_labels_list = [
            l for l, s in zip(all_labels, all_signer_ids) if s == test_signer
        ]
        test_ds = create_tf_dataset(
            test_paths,
            test_labels_list,
            label_encoder,
            signer_info["global_mean"],
            signer_info["global_std"],
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            augment=False,
        )
        print(
            f"  Train/val signers: {len(train_val_signers)} | "
            f"Test: {test_signer} ({len(test_paths)} samples)"
        )

        input_dim = MODEL_INPUT_DIM
        CONFIG["num_classes"] = num_classes

        model = build_mobile_sign_gru(
            input_dim=input_dim,
            num_classes=num_classes,
            max_len=CONFIG["max_len"],
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
            bidirectional=CONFIG["bidirectional"],
            l2_reg=CONFIG.get("l2_reg", 1e-3),
            conv_filters=CONFIG.get("conv_filters", [128, 128]),
            conv_kernel_size=CONFIG.get("conv_kernel_size", 5),
            spatial_dropout=CONFIG.get("spatial_dropout", 0.2),
            recurrent_dropout=CONFIG.get("recurrent_dropout", 0.2),
            use_mask_concat=CONFIG.get("use_mask_concat", True),
        )
        if fold == 0:
            model.summary()
            param_count = model.count_params()
            model_size_mb = sum(w.numpy().nbytes for w in model.weights) / (1024 * 1024)
            print(f"Parameters: {param_count:,}  Size: {model_size_mb:.2f} MB")

        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        if steps_per_epoch < 0:
            n_train = len(
                [
                    p
                    for p, s in zip(all_paths, all_signer_ids)
                    if s != test_signer and s not in signer_info["val_signers"]
                ]
            )
            steps_per_epoch = max(1, n_train // CONFIG["batch_size"])

        val_steps = tf.data.experimental.cardinality(val_ds).numpy()
        if val_steps < 0:
            val_steps = None

        model, history_obj = train_tf_model(
            train_ds,
            val_ds,
            CONFIG,
            num_classes,
            input_dim,
            steps_per_epoch,
            val_steps,
        )

        test_loss, test_acc = model.evaluate(test_ds)
        eval_results = evaluate_model(model, test_ds, label_encoder)
        print(f"  >>> Fold {fold + 1} test accuracy: {test_acc:.4f}")

        fold_results.append(
            {
                "test_signer": test_signer,
                "test_accuracy": float(test_acc),
                "val_accuracy": float(
                    max(history_obj.history.get("val_accuracy", [0]))
                ),
            }
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model
            best_fold_idx = fold
            best_eval_results = eval_results
            best_history_obj = history_obj
            best_test_signer = test_signer

    test_accs = [r["test_accuracy"] for r in fold_results]
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS ({N} folds)")
    print(f"{'=' * 60}")
    for r in fold_results:
        print(f"  {r['test_signer']:<12}  test_acc={r['test_accuracy']:.4f}")
    print(f"  {'─' * 40}")
    print(f"  Mean:   {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"  Median: {np.median(test_accs):.4f}")
    print(f"  Best:   {best_test_signer} ({best_test_acc:.4f})")

    print(f"\n{'=' * 60}")
    print(f"Exporting best model (fold {best_fold_idx + 1})")
    print(f"{'=' * 60}")

    model = best_model
    history_obj = best_history_obj
    eval_results = best_eval_results

    np.save(OUTPUT_DIR / "history.npy", history_obj.history)

    param_count = model.count_params()
    model_size_mb = sum(w.numpy().nbytes for w in model.weights) / (1024 * 1024)

    CONFIG_for_export = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dim": CONFIG["hidden_dim"],
        "num_layers": CONFIG["num_layers"],
        "dropout": CONFIG["dropout"],
        "bidirectional": CONFIG["bidirectional"],
        "max_len": CONFIG["max_len"],
        "l2_reg": CONFIG.get("l2_reg", 1e-3),
        "conv_filters": CONFIG.get("conv_filters", [128, 128]),
        "conv_kernel_size": CONFIG.get("conv_kernel_size", 5),
        "use_mask_concat": CONFIG.get("use_mask_concat", True),
        "spatial_dropout": CONFIG.get("spatial_dropout", 0.2),
        "recurrent_dropout": CONFIG.get("recurrent_dropout", 0.2),
        "num_params": int(param_count),
        "model_size_mb": round(model_size_mb, 3),
        "label_classes": list(label_encoder.classes_),
        "cross_val_accuracy_mean": float(np.mean(test_accs)),
        "cross_val_accuracy_std": float(np.std(test_accs)),
        "best_test_signer": best_test_signer,
        "best_test_accuracy": float(best_test_acc),
    }

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CONFIG_for_export, f, indent=2)

    plot_training_history(history_obj, OUTPUT_DIR)
    plot_confusion_matrix(eval_results, OUTPUT_DIR)
    save_evaluation_results(eval_results, OUTPUT_DIR)

    print("\nBenchmarking TF model...")
    bench_tf = benchmark_tf_model(model, input_dim, CONFIG["max_len"])
    print(f"  {bench_tf['ms_per_sample']:.3f} ms/sample ({bench_tf['fps']:.1f} fps)")

    print("\nExporting to TFLite...")
    saved_model_path = OUTPUT_DIR / "tf_saved_model"
    model.export(saved_model_path)
    print(f"  SavedModel → {saved_model_path}")

    tflite_path = OUTPUT_DIR / "model.tflite"
    convert_saved_model_to_tflite(saved_model_path, tflite_path)

    print("\nExporting self-contained TFLite (raw landmarks)...")
    export_selfcontained_tflite(
        model=model,
        data_paths=all_paths,
        config=CONFIG_for_export,
        output_dir=OUTPUT_DIR,
    )

    print("\nBenchmarking TFLite...")
    tflite_bench = benchmark_tflite_model(tflite_path)
    if tflite_bench is not None:
        print(
            f"  {tflite_bench['mean_ms']:.3f} ± {tflite_bench['std_ms']:.3f} ms "
            f"({tflite_bench['fps']:.1f} fps)"
        )

    print(f"\n{'=' * 60}")
    print("BISINDO SIGN LANGUAGE RECOGNITION — PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(
        f"\nFinal cross-val accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}"
    )
    print(
        f"Best model (fold {best_fold_idx + 1}): {best_test_signer} = {best_test_acc:.4f}"
    )
    print(f"\nOutput files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
