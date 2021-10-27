import numpy as np
import keras.backend as K
from keras.models import load_model

model = load_model('models/30_model_andrew.h5')

def center_norm(track, length=30):
    """
    Returns track steps after transforming to unit
    variance and 0 mean shift
    """
    track -= np.mean(track)
    dx = np.diff(track)
    dx = dx[1:length]
    variance = np.std(dx)
    dx /= variance
    dx = np.reshape(dx, (1, 29, 1))
    return dx

def get_cam(track):
    # Get the layer of the last conv layer
    final_conv_layer  = model.get_layer('conv1d_12')
    # Get the weights matrix of the last layer
    class_weights = model.layers[-1].get_weights()[0]
    # Prepare track
    #     track =np.loadtxt("em18tracks.txt")[i]

    t = center_norm(track)

    # # Print what the top predicted class is
    index_to_type = {0: "fbm", 1: "brownian", 2: "ctrw"}
    preds = model.predict(t)
    c = np.argmax(preds)
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([t])
    conv_outputs = conv_outputs[0, ...]
    cam = class_weights[np.newaxis, :, c] @ conv_outputs.T
    return cam
