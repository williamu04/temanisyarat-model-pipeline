import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def compute_global_stats(paths, max_len):
    from src.data import extract_features_with_mask

    all_feats, all_masks = [], []
    for path in paths:
        feat, mask = extract_features_with_mask(path)
        T = min(len(feat), max_len)
        all_feats.append(feat[:T])
        all_masks.append(mask[:T])
    all_feats = np.concatenate(all_feats, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    masked_sum = np.sum(all_feats * all_masks, axis=0, keepdims=True)
    masked_count = np.sum(all_masks, axis=0, keepdims=True) + 1e-8
    mean = (masked_sum / masked_count).astype(np.float32)
    diff = (all_feats - mean) * all_masks
    masked_var = np.sum(diff**2, axis=0, keepdims=True) / masked_count
    std = (np.sqrt(masked_var) + 1e-8).astype(np.float32)
    return mean, std


def load_weights_from_savedmodel(model, saved_model_dir):
    imported = tf.saved_model.load(str(saved_model_dir))
    src_map = {}
    for v in imported.variables:
        name = v.name[:-2] if v.name.endswith(":0") else v.name
        src_map[name] = v

    unmatched = []
    for dst_var in model.weights:
        p = dst_var.path
        if p in src_map:
            dst_var.assign(src_map[p].read_value())
            continue
        # Try matching with any _N suffix (e.g. dense_5/kernel -> dense/kernel)
        matched = False
        for src_name in src_map:
            # Check if src_name = base_N/rest and dst = base/rest
            src_parts = src_name.rsplit("/", 1)
            if len(src_parts) == 2:
                src_base, src_rest = src_parts
                dst_base, dst_rest = p.rsplit("/", 1)
                if src_rest == dst_rest and src_base.startswith(dst_base + "_"):
                    suffix = src_base[len(dst_base) :]
                    if suffix.lstrip("_").isdigit():
                        dst_var.assign(src_map[src_name].read_value())
                        matched = True
                        break
        if not matched:
            unmatched.append(p)

    if unmatched:
        print(f"  WARNING: {len(unmatched)} variables not matched:")
        for u in unmatched:
            print(f"    {u}")
    else:
        print("  All weights loaded successfully.")
    return len(model.weights) - len(unmatched)


def export_selfcontained_tflite(
    model,
    data_paths,
    config,
    output_dir,
    saved_model_dir="output/tf_saved_model",
):
    """Wrap a trained model with raw-input preprocessing and export to TFLite.

    The resulting model accepts raw (125, 153) landmarks (with NaNs for
    undetected keypoints) and handles NaN detection, masking, and
    normalization internally -- no external preprocessing needed.
    Uses a custom RawPreprocessor layer with TFLite-native ops.
    """
    output_dir = Path(output_dir)
    max_len = int(config["max_len"])
    raw_dim = int(config["input_dim"]) // 2  # 153

    # ---- Compute normalization stats ----
    print("Computing global mean/std from dataset...")
    global_mean, global_std = compute_global_stats(data_paths, max_len)
    np.save(output_dir / "global_mean.npy", global_mean)
    np.save(output_dir / "global_std.npy", global_std)
    print(f"  Mean shape: {global_mean.shape}, Std shape: {global_std.shape}")

    # ---- Load/reuse model ----
    if model is None:
        print("Rebuilding model from SavedModel...")
        from src.model import build_mobile_sign_gru

        model = build_mobile_sign_gru(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            max_len=max_len,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            l2_reg=config.get("l2_reg", 1e-3),
            conv_filters=config.get("conv_filters", [128, 128]),
            conv_kernel_size=config.get("conv_kernel_size", 5),
            spatial_dropout=config.get("spatial_dropout", 0.2),
            recurrent_dropout=config.get("recurrent_dropout", 0.2),
            use_mask_concat=config.get("use_mask_concat", True),
        )
        model(
            np.random.randn(1, max_len, config["input_dim"]).astype(np.float32),
            training=False,
        )
        load_weights_from_savedmodel(model, saved_model_dir)

    # ---- Build wrapper with matching normalization ----
    print("Building self-contained wrapper model...")
    raw_input = tf.keras.Input(
        shape=(max_len, raw_dim), dtype="float32", name="raw_input"
    )
    m = global_mean.reshape(1, 1, -1).astype(np.float32)
    s = global_std.reshape(1, 1, -1).astype(np.float32)

    # Step 1: valid_mask = 1 for valid landmarks, 0 for NaN/padding
    valid_mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.logical_not(tf.math.is_nan(x)), tf.float32),
        name="valid_mask",
    )(raw_input)
    # Step 2: NaN → 0, then normalise ALL values
    normalized = tf.keras.layers.Lambda(
        lambda x, m_val=tf.constant(m), s_val=tf.constant(s): (
            (tf.where(tf.math.is_nan(x), tf.zeros_like(x), x) - m_val) / (s_val + 1e-8)
        ),
        name="normalize",
    )(raw_input)
    # Step 3: zero-out NaN/padding positions after normalisation
    masked_normalized = tf.keras.layers.Lambda(
        lambda x: x[0] * x[1],
        name="apply_mask",
    )([normalized, valid_mask])
    # Step 4: concat features + mask
    preprocessed = tf.keras.layers.Concatenate(axis=-1, name="feat_mask_concat")(
        [masked_normalized, valid_mask]
    )
    outputs = model(preprocessed)
    wrapper = tf.keras.Model(raw_input, outputs, name="mobilesign_gru_raw")

    # ---- Convert to TFLite (float32, no quantization) ----
    converter = tf.lite.TFLiteConverter.from_keras_model(wrapper)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    tflite_path = output_dir / "model_raw.tflite"
    tflite_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"  Saved: {tflite_path} ({size_mb:.2f} MB)")

    print("  Done! Model accepts raw (125, 153) landmarks with NaNs.")
    return tflite_path, size_mb


def convert_saved_model_to_tflite(saved_model_dir, output_path):
    """Convert TensorFlow SavedModel to TFLite format."""
    print("Converting to TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    try:
        tflite_model = converter.convert()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"TFLite model saved to {output_path} ({size_mb:.2f} MB)")
        return
    except Exception as e:
        print(f"TFLite Conversion failed: {e}")
        raise e


def benchmark_tf_model(model, input_dim, max_len, n_runs=100):
    """Benchmark TF model inference speed."""
    dummy = tf.random.normal((1, max_len, input_dim))

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(dummy, training=False)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "ms_per_sample": round(float(np.mean(times)), 3),
        "fps": round(1000 / float(np.mean(times)), 1),
    }


def benchmark_tflite_model(tflite_path, n_runs=100):
    """Benchmark TFLite model inference."""
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
    except RuntimeError as e:
        print(
            f"Skipping TFLite benchmark. The model requires Flex delegate for SELECT_TF_OPS: {
                e
            }"
        )
        return None

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    print(f"Detected TFLite input shape: {input_shape}")

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": round(np.mean(times), 3),
        "std_ms": round(np.std(times), 3),
        "fps": round(1000 / np.mean(times), 1),
    }
