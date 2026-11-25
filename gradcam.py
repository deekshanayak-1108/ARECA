import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

MODEL_PATH = "areca_mobilenetv2.keras"
LAST_CONV_LAYER_NAME = "Conv_1"

CLASS_NAMES = [
    "anthracnose",
    "bacterial_blotch",
    "fruit_split",
    "healthy",
    "mahali_koleroga",
]


def generate_gradcam(
    image_path,
    output_path="static/gradcam_result.jpg",
    model=None,
    model_path=MODEL_PATH,
    class_names=CLASS_NAMES,
    last_conv_layer_name=LAST_CONV_LAYER_NAME,
    pred_index=None,
    alpha=0.4,
):
    if model is None:
        model = load_model(model_path)

    img = load_img(image_path, target_size=(224, 224))
    original_rgb = img_to_array(img).astype("uint8")
    preprocessed = np.expand_dims(original_rgb, axis=0) / 255.0

    base_model = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model)),
        None,
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(preprocessed, tf.float32)

        if base_model is not None:
            last_conv_layer = base_model.get_layer(last_conv_layer_name)
            base_grad_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=[last_conv_layer.output, base_model.output],
            )
            conv_outputs, x = base_grad_model(inputs)
            tape.watch(conv_outputs)

            passed_base = False
            for layer in model.layers:
                if layer is base_model:
                    passed_base = True
                    continue
                if passed_base:
                    x = layer(x)
            preds = x
        else:
            last_conv_layer = model.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model.output],
            )
            conv_outputs, preds = grad_model(inputs)
            tape.watch(conv_outputs)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    heatmap_resized = cv2.resize(
        heatmap_np, (original_rgb.shape[1], original_rgb.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(heatmap_colored_rgb * alpha + original_rgb * (1 - alpha))
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, overlay_bgr)

    probs = preds[0].numpy()
    all_scores = {
        class_names[i]: float(probs[i] * 100)
        for i in range(min(len(class_names), len(probs)))
    }
    predicted_label = (
        class_names[pred_index] if pred_index < len(class_names) else str(pred_index)
    )

    return {
        "predicted_class": predicted_label,
        "confidence": float(probs[pred_index] * 100),
        "all_scores": all_scores,
        "output_path": output_path,
    }


if __name__ == "__main__":
    result = generate_gradcam(image_path="61.jpg", output_path="gradcam_61.jpg")
    print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2f}%)")
    print("All class scores:")
    for cls, score in result["all_scores"].items():
        print(f"  - {cls}: {score:.2f}%")
    print(f"Saved to: {result['output_path']}")