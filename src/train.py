import tensorflow as tf


def train_tf_model(
    train_ds,
    val_ds,
    CONFIG,
    num_classes,
    input_dim,
    steps_per_epoch=None,
    validation_steps=None,
):
    from .model import build_mobile_sign_gru

    model = build_mobile_sign_gru(
        input_dim,
        num_classes,
        CONFIG["max_len"],
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

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CONFIG["learning_rate"],
        clipnorm=CONFIG.get("gradient_clip_norm", 1.0),
    )

    cce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=CONFIG.get("label_smoothing", 0.1),
    )

    def loss_fn(y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        return cce(y_true_one_hot, y_pred)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=CONFIG["patience"],
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=CONFIG["patience"],
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    for x, y in train_ds.take(1):
        print(x.shape, y.shape)

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
    )

    return model, history
