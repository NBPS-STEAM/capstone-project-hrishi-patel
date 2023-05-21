import gradio
import gradio as gr
import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('digits.model')


def classify(input):
    input = np.reshape(input, (1, 28, 28))
    prediction = new_model.predict(input).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

label = gr.outputs.Label(num_top_classes=10)
sp = gradio.Sketchpad(image_mode="L", source="canvas", shape=(14, 14), invert_colors=False, interactive=True)

interface = gr.Interface(fn=classify, inputs="sketchpad", outputs=label,
live=True)
interface.launch()
