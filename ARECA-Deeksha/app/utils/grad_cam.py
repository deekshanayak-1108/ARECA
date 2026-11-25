import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_img_array(img_path, size=(224, 224)):
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(array, axis=0)


def generate_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    img_tensor = tf.cast(img_array, tf.float32)

    if hasattr(model, "layers") and isinstance(model.layers[0], tf.keras.Model):
        base_model = model.layers[0]
        conv_layer = base_model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[conv_layer.output, base_model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(img_tensor)
            tape.watch(conv_outputs)
            x = base_outputs
            for layer in model.layers[1:]:
                x = layer(x)
            if pred_index is None:
                pred_index = tf.argmax(x[0])
            class_channel = x[:, pred_index]
    else:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_tensor)
            tape.watch(conv_outputs)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4, output_path=None):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    colormap = plt.get_cmap("jet")
    heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    superimposed_rgb = np.clip(heatmap_colored * alpha + img_rgb * (1 - alpha), 0, 255).astype(np.uint8)
    superimposed_bgr = cv2.cvtColor(superimposed_rgb, cv2.COLOR_RGB2BGR)

    if output_path:
        cv2.imwrite(output_path, superimposed_bgr)

    return superimposed_bgr


def generate_gradcam(img_path, model, last_conv_layer_name="Conv_1", output_path=None):
    img_array = get_img_array(img_path)
    heatmap = generate_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name)
    return overlay_gradcam(img_path, heatmap, alpha=0.4, output_path=output_path)