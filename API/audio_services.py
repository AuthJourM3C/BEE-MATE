# import libraries
import numpy as np
from flask import Flask, request, render_template
import librosa
import os, json, warnings
from waitress import serve
from mutagen import mp3, wave
from PIL import Image, ImageStat
import imquality.brisque as brisque
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from image_captioning import feature_extractions, sample_caption
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import tokenizer_from_json
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


# ***********************
# *** AUDIO ENDPOINTS ***
# ***********************

# audio information
def audio_info():
    if request.method == 'POST':
        # prepare file
        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)

        quality = 3

        # read file
        if file.filename.find('mp3') > 0:
            f = mp3.MP3(file)
            format = 'mp3'
            sample_rate = f.info.sample_rate
            bitdepth = 'N/A'
            bitrate = {f.info.bitrate / 1000}
            quality = (bitrate / 48) * (sample_rate / 44100)
            channels = f.info.channels
            length = f.info.length

        if file.filename.find('wav') > 0:
            f = wave.WAVE(file)
            format = 'wav'
            sample_rate = f.info.sample_rate
            bitdepth = f.info.bits_per_sample
            bitrate = 'N/A'
            quality = min((bitdepth / 16), 1.5) * min((sample_rate / 11025), 5)
            channels = f.info.channels
            length = f.info.length

        quality = int(round(quality))
        quality = min(max(quality, 1), 5)

        # delete file
        os.remove(filepath)

        # respond
        response = {
            "format": format,
            "samplerate": sample_rate,
            "bitdepth": f"""{bitdepth}""",
            "bitrate": f"""{bitrate} kbps""",
            "length": f"""{length:.1f} s""",
            "channels": f"""{channels}""",
            "quality": f"""{quality}/5."""
        }
        # response = json.dumps(response)
        # response = f"""Audio file format is {format}, sample rate is {sample_rate}, bitrate is {bitrate:.0f}, length is {length:.1f}s and number of channels is {channels}."""
        # print(response)
        return response