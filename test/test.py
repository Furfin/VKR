import keras
import autokeras as ak
import matplotlib.pyplot as plt

dot_img_file = '.model_1.png'

model =  keras.saving.load_model('model1.keras')
print(model.layers)
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model_info = {
        "name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "trainable_params": model.count_params(),
        "layers": [],
        "config": model.get_config()
    }
    
# Extract layer information
for layer in model.layers:
    layer_info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "input_shape": layer.input_spec,
        "trainable": layer.trainable,
        "params": layer.count_params(),
        "config": layer.get_config()
    }
    model_info["layers"].append(layer_info)
    
print(model_info)