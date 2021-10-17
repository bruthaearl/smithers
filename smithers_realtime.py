# %%
import sounddevice as sd
import numpy as np
import python_speech_features
from tflite_runtime.interpreter import Interpreter

model_path = 'smithers_lite.tflite'
rec_duration = 0.5
sample_rate = 10000
debug = 0
threshold = 0.5

# %%
window = np.int16(np.zeros(int(rec_duration * sample_rate) * 2))
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# %%
# stream callback
def sd_callback(rec, frames, time, status):
   
    # remove unnecessary dimension
    rec = np.squeeze(rec)

    # sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # calculate mfccs
    mfccs = python_speech_features.base.mfcc(window, samplerate=sample_rate, winstep=0.05, numcep=13)
    mfccs = mfccs.transpose()

    # set up tensors
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))

    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    val = output[0][0]
    if val > threshold:
        print('smithers')

    if debug:
        print(val)

# %%
# start stream
with sd.InputStream(channels=1,
                    samplerate=sample_rate,
                    dtype='int16',
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
