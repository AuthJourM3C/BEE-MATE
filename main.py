# conda venv
# conda create -n socio_bee python=3.10
# conda install -c anaconda flask
# conda install -c conda-forge waitress
# conda install -c conda-forge librosa
# conda install -c conda-forge tensorflow
# conda install -c conda-forge mutagen
# conda install -c conda-forge image-quality
# pip install pyAudioAnalysis

# import libraries
#import datetime

import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import librosa
import os, json, warnings
from waitress import serve
from mutagen import mp3, wave
from PIL import Image, ImageStat
import imquality.brisque as brisque
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from image_captioning import feature_extractions, sample_caption
from datetime import datetime
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import tokenizer_from_json
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from collections import Counter
from tinytag import TinyTag


# declare global variables
global model_3_class_smo_cnn1d, model_2_class_pollution_cnn1d, model_image_classification, model_image_captioning, tokenizer_image_captioning

# initialise flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = '%\\\xdb\xe1\x99\xec\xfb\xefU\xeb\x11Gv\xac}\x92'
CORS(app)

# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    return response

# ***********************
# *** AUDIO ENDPOINTS ***
# ***********************

# audio information
@app.route('/api/audio/info', methods=['GET', 'POST'])
def audio_info(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file if needed
        if filepath_alt == '':
            #print(request.files['messageFile'])
            file = request.files['messageFile']
            print(file)
            filepath = './temp/' + file.filename
            file.save(filepath)
            #print("filename:" + file.filename)
        else:
            file = open(filepath_alt,"rb")
            filepath = filepath_alt
            print(filepath)

        # read file
        if filepath.find('mp3') > 0:
            # f=mp3.MP3(filepath_alt)
            f = mp3.MP3(file)
            format = 'mp3'
            sample_rate = f.info.sample_rate
            bitdepth = 'N/A'
            bitrate = f"""{f.info.bitrate / 1000} kbps"""
            channels = f.info.channels
            length = f.info.length
        if filepath.find('wav') > 0:
            f = wave.WAVE(file)
            format = 'wav'
            sample_rate = f.info.sample_rate
            bitdepth = f"""{f.info.bits_per_sample}"""
            bitrate = 'N/A'
            channels = f.info.channels
            length = f.info.length

        # delete file if needed
        if filepath_alt == '':
            os.remove(filepath)

        # respond
        response = {
            "format": format,
            "samplerate": sample_rate,
            "bitdepth": bitdepth,
            "bitrate": bitrate,
            "length": f"""{length:.1f} s""",
            "channels": f"""{channels}""",
            "timestamp": datetime.now(),
            "location": {
                "Lat": "N/A",
                "Lon": "N/A"
            }
        }
        # response = json.dumps(response)

        # response = f"""Audio file format is {format}, sample rate is {sample_rate}, bitrate is {bitrate:.0f}, length is {length:.1f}s and number of channels is {channels}."""
        # print(response)
        return response

# audio quality
@app.route('/api/audio/quality', methods=['GET', 'POST'])
def audio_quality(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file if needed
        if filepath_alt == '':
            file = request.files['messageFile']
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            file = open(filepath_alt, "rb")
            filepath = filepath_alt

        quality = 3
        if filepath.find('mp3') > 0:
            f = mp3.MP3(file)
            sample_rate = f.info.sample_rate
            bitrate = f.info.bitrate / 1000
            quality = (bitrate/48) * (sample_rate/44100)
            format = "mp3"
        if filepath.find('wav') > 0:
            f = wave.WAVE(file)
            sample_rate = f.info.sample_rate
            bitrate = f.info.bits_per_sample
            quality = min((bitrate/16),1.5) * min((sample_rate/11025),5)
            format = "wav"

        quality = int(round(quality))
        quality = min(max(quality, 1), 5)

        # delete file if needed
        if filepath_alt == '':
            os.remove(filepath)

        # response = f"""Audio file quality is {quality}/5."""
        response = {
            "format" : format,
            "quality": quality
        }
        return response

# audio features
@app.route('/api/audio/features', methods=['GET', 'POST'])
def audio_features(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':

        # prepare file if needed
        if filepath_alt == '':
            file = request.files['messageFile']
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        sr = 22050
        block_size = int(1 * sr)
        step_size = int(0.5 * sr)
        audiofile, rate = librosa.load(filepath, sr=sr, mono=True, dtype=float)
        # marray = np.ones((int(audiofile.shape[0]/step_size), block_size, 1))

        # delete file if needed
        if filepath_alt == '':
            os.remove(filepath)

        # length = audiofile.shape[0]/22050
        # for i in range(0, audiofile.shape[0] - block_size, step_size):
            #
            # marray[int(i/step_size),:,0] = audiofile[i:i+block_size] / np.sqrt(np.mean(np.square(audiofile[i:i+block_size])))
        # zcr = librosa.feature.rms(marray)
        # print(marray[0,:,0])

        rms = librosa.feature.rms(y=audiofile)
        zcr = librosa.feature.zero_crossing_rate(y=audiofile)
        s_cent = librosa.feature.melspectrogram(y=audiofile, sr=rate)
        mel = librosa.feature.spectral_centroid(y=audiofile, sr=rate)
        mfcc = librosa.feature.mfcc(y=audiofile, sr=rate)
        chrom = librosa.feature.chroma_stft(y=audiofile, sr=rate)
        s_bw = librosa.feature.spectral_bandwidth(y=audiofile, sr=rate)
        s_cont = librosa.feature.spectral_contrast(y=audiofile, sr=rate)
        s_flat = librosa.feature.spectral_flatness(y=audiofile)
        s_rol = librosa.feature.spectral_rolloff(y=audiofile, sr=rate)

        # respond
        response = {
            "rms": str(rms[0]).replace("\n", ""),
            "zcr": str(zcr[0]).replace("\n", ""),
            "mel": str(mel).replace("\n", ""),
            "mfcc": str(mfcc).replace("\n", ""),
            "chrom": str(chrom).replace("\n", ""),
            "s_cent": str(s_cent[0]).replace("\n", ""),
            "s_bw": str(s_bw[0]).replace("\n", ""),
            "s_cont": str(s_cont[0]).replace("\n", ""),
            "s_flat": str(s_flat[0]).replace("\n", ""),
            "s_rol": str(s_rol[0]).replace("\n", ""),
        }
        return response

# audio classification
@app.route('/api/audio/classification', methods=['GET', 'POST'])
def audio_classification(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file if needed
        if filepath_alt == '':
            file = request.files['messageFile']
            # print(type(file))
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        sr = 22050
        block_size = int(1 * sr)
        step_size = int(0.5 * sr)
        audiofile, rate = librosa.load(filepath, sr=sr, mono=True, dtype=float)
        marray = np.ones((int(audiofile.shape[0]/step_size), block_size, 1))

        if filepath_alt == '':
            os.remove(filepath)

        length = audiofile.shape[0]/22050
        for i in range(0, audiofile.shape[0] - block_size, step_size):
            #
            marray[int(i/step_size),:,0] = audiofile[i:i+block_size] / np.sqrt(np.mean(np.square(audiofile[i:i+block_size])))

        #transfer from pollution
        ESC_50_classes = ['dog','rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insect', 'sheep', 'crow',
                          'rain', 'sea waves', 'fire', 'crickets', 'chirping birds', 'water drops', 'wind',
                          'pouring water', 'toilet flush', 'thunderstorm', 'crying baby', 'sneezing',
                          'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing teeth',
                          'snoring', 'drinking-sipping', 'door knock', 'mouse click', 'keyboard typing',
                          'door creeak', 'can oppening', 'washing machine', 'vacuum cleaner', 'clock alarm',
                          'clock tick', 'glass breaking', 'hellicopter', 'chainsaw', 'siren', 'car horn',
                          'engine', 'train', 'church bells', 'airplane', 'fireworks', 'hand saw']

        prediction = model_3_class_smo_cnn1d.predict(marray)
        prediction = np.argmax(prediction, axis=1)
        alt_prediction = model_cnn1d_esc_50.predict(marray)
        alt_prediction = np.argmax(alt_prediction, axis=1)
        text = []
        for i in range(0, len(prediction), 1):
            text.append('{:.1f}s to {:.1f}s:'.format(i * step_size / sr, i * step_size / sr + block_size / sr))
            # if prediction[i] == 0: text.append('music')
            if prediction[i] == 0: text.append('rythmic pattern')
            if prediction[i] == 1: text.append('speech')
            if prediction[i] == 2:
                text.append(ESC_50_classes[alt_prediction[i]])

        # prediction = model_3_class_smo_cnn1d.predict(marray)
        # prediction = model_cnn1d_esc_50.predict(marray)
        # prediction = np.argmax(prediction, axis=1)
        # text = []
        #
        # ESC_50_classes = ['dog','rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insect', 'sheep', 'crow', 'rain', 'sea waves', 'fire', 'crickets', 'chirping birds', 'water drops', 'wind', 'pouring water', 'toilet flush', 'thunderstorm', 'crying baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing teeth', 'snoring', 'drinking-sipping', 'door knock', 'mouse click', 'keyboard typing', 'door creeak', 'can oppening', 'washing machine', 'vacuum cleaner', 'clock alarm', 'clock tick', 'glass breaking', 'hellicopter', 'chainsaw', 'siren', 'car horn', 'engine', 'train', 'church bells', 'airplane', 'fireworks', 'hand saw']
        # for i in range(0, len(prediction), 1):
        #     text.append('{:.1f}s to {:.1f}s:'.format(i*step_size/sr, i*step_size/sr+block_size/sr))
        #     text.append(ESC_50_classes[prediction[i]])

        # label decoding for smo
        # for i in range(0, len(prediction), 1):
        #     text.append('{:.1f}s to {:.1f}s:'.format(i*step_size/sr, i*step_size/sr+block_size/sr))
        #     if prediction[i] == 0: text.append('music')
        #     if prediction[i] == 1: text.append('speech')
        #     if prediction[i] == 2: text.append('others')

        # response = f"""Audio file prediction is \"{text}\"."""
        # print(response)
        # respond
        response = {
            "audioclass": text,
        }
        return response

# audio pollution/non pollution classification
@app.route('/api/audio/pollution', methods=['GET', 'POST'])
def audio_polluting(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file if needed
        if filepath_alt == '':
            file = request.files['messageFile']
            # print(type(file))
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        sr = 22050
        block_size = int(1 * sr)
        step_size = int(0.5 * sr)
        audiofile, rate = librosa.load(filepath, sr=sr, mono=True, dtype=float)
        marray = np.ones((int(audiofile.shape[0]/step_size), block_size, 1))

        if filepath_alt == '':
            os.remove(filepath)

        length = audiofile.shape[0]/22050
        for i in range(0, audiofile.shape[0] - block_size, step_size):
            marray[int(i/step_size),:,0] = audiofile[i:i+block_size] / np.sqrt(np.mean(np.square(audiofile[i:i+block_size])))

        prediction = model_3_class_smo_cnn1d.predict(marray)
        prediction = np.argmax(prediction, axis=1)
        alt_prediction = model_2_class_pollution_cnn1d.predict(marray)
        alt_prediction = np.argmax(alt_prediction, axis=1)
        text = []
        for i in range(0, len(prediction), 1):
            text.append('{:.1f}s to {:.1f}s:'.format(i*step_size/sr, i*step_size/sr+block_size/sr))
            if prediction[i] == 0: text.append('music')
            if prediction[i] == 1: text.append('speech')
            if prediction[i] == 2:
                if alt_prediction[i] == 0: text.append('others')
                if alt_prediction[i] == 1: text.append('others - polluting')

        # response = f"""Audio file prediction is \"{text}\"."""
        # print(response)
        # respond
        response = {
            "audiopollutionclass": text,
        }
        return response

# audio speech to text
@app.route('/api/audio/stt', methods=['GET', 'POST'])
def speechtotext():
    if request.method == 'POST':
        response = ''
        return response

#audio_academe_api
@app.route('/api/audio/academe', methods=['POST'])
def audio_academe():
    file = request.files['messageFile']
    # uploaded_file = request.files['messageFile']
    if 'messageFile' not in request.files:
        problem = "messageFile not found in request.files"
        return 'problem'
    else:
        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)

    audio_classification_response = audio_classification (filepath)
    audio_pollution_response = audio_polluting (filepath)

    # classes = list(audio_classification_response.values())
    # classes = [item for sublist in classes for item in sublist]
    # classes = [x for x in classes if 's to ' not in x]
    # classes = list(dict.fromkeys(classes))
    # print (classes)
    # print (type(classes))

    classes = list(audio_classification_response.values())
    classes = [item for sublist in classes for item in sublist]
    classes = [x for x in classes if 's to ' not in x]
    counter = Counter(classes)

    counter = {k:v/len(classes) for k,v in counter.items()}

    # response = {**audio_classification_response, **audio_pollution_response}

    audiopollutionclass = "not polluting"
    #METHOD1 -  audio_pollution binary classification model
    # pollution = list(audio_pollution_response.values())
    # pollution = [item for sublist in pollution for item in sublist]
    # print (pollution)
    # if "others - polluting" in pollution:
    #     audiopollutionclass = "polluting"

    #METHOD2 - audio classfication model - dictionary
    ESC_50_pollution_classes = ['washing machine', 'vacuum cleaner', 'hellicopter', 'chainsaw', 'siren', 'car horn',
                                'engine', 'train', 'airplane']
    for audio_class in list(counter.keys()):
        if audio_class in ESC_50_pollution_classes:
            audiopollutionclass = "polluting"
            pollutioninfo = "Source of pollution is detected!"

    response = {"audioclassification":counter, "audiopollutionclass":audiopollutionclass}
    # delete file
    os.remove(filepath)
    return response


#audio_captioning
@app.route('/api/audio/captioning', methods=['POST'])
def audio_captioning():
    file = request.files['messageFile']
    # uploaded_file = request.files['messageFile']
    if 'messageFile' not in request.files:
        problem = "messageFile not found in request.files"
        return 'problem'
    else:
        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)

    audio_classification_response = audio_classification (filepath)
    audio_pollution_response = audio_polluting (filepath)

    classes = list(audio_classification_response.values())
    classes = [item for sublist in classes for item in sublist]
    classes = [x for x in classes if 's to ' not in x]
    counter = Counter(classes)

    counter = {k:v/len(classes) for k,v in counter.items()}

    # response = {**audio_classification_response, **audio_pollution_response}

    audiopollutionclass = "not polluting"

    ESC_50_pollution_classes = ['washing machine', 'vacuum cleaner', 'hellicopter', 'chainsaw', 'siren', 'car horn',
                                'engine', 'train', 'airplane']
    detected_pollution_classes = []
    for audio_class in list(counter.keys()):
        if audio_class in ESC_50_pollution_classes:
            audiopollutionclass = "polluting"
            pollutioninfo = "Source of pollution is detected!"
            detected_pollution_classes.append(audio_class)
    # print (counter)
    # print (counter.keys())
    # print (list(counter.keys()))
    # print(list(counter.keys())[0])
    # print(list(counter.values())[0])
    # pollution_classes_str = str(detected_pollution_classes).replace("'", "")

    detected_classes_str = ','.join(list(counter.keys()))
    detected_pollution_classes_str = ', '.join(detected_pollution_classes)
    if len(detected_pollution_classes)>1:
        response = 'The following audio classes have been detected: ' + detected_classes_str + ', of which, ' \
                   + detected_pollution_classes_str + ' are considered to be related to air pollution events.'
    elif len(detected_pollution_classes)==1:
        response = 'The following audio classes have been detected: ' + detected_classes_str + ', of which, ' \
                   + detected_pollution_classes_str + ' is considered to be related to air pollution events.'
    else:
        response = 'The following audio classes have been detected: ' + detected_classes_str + ', of which, ' \
                   'none is considered to be related to air pollution events.'

    # print ('The following classes have been detected:' + str.(counter.keys()) + 'The dominant class is "' + list(counter.keys())[0]+ 'and the following classes related to pollution have been detected:' + detected_pollution_classes)
    # response = 'The following classes have been detected:' + str.(counter.keys()) + 'The dominant class is '
    # response = {"audioclassification":counter, "audiopollutionclass":audiopollutionclass}
    # delete file
    os.remove(filepath)
    return response

#Audio-driven Bee-MATE
@app.route('/api/audio/beemate', methods=['POST'])
def bee_mate(filepath_alt=''):
    print("test")
    if filepath_alt == '':
        print("if")
        file = request.files['messageFile']
        #print(file)
        filepath = './temp/' + file.filename
        file.save(filepath)
    else:
        print("else")
        file = open(filepath_alt,"rb")
        filepath = filepath_alt
        print(filepath)

    #classification
    audio_classification_response = audio_classification(filepath)
    audio_pollution_response = audio_polluting(filepath)
    classes = list(audio_classification_response.values())
    classes = [item for sublist in classes for item in sublist]
    classes = [x for x in classes if 's to ' not in x]
    counter = Counter(classes)

    counter = {k: v / len(classes) for k, v in counter.items()}

    audio_pollution_class = "No pollution source detected."
    pollution_source = "no_pollution"
    # METHOD1 -  audio_pollution binary classification model
    # pollution = list(audio_pollution_response.values())
    # pollution = [item for sublist in pollution for item in sublist]
    # print (pollution)
    # if "others - polluting" in pollution:
    #     audiopollutionclass = "polluting"

    # METHOD2 - audio classfication model - dictionary
    ESC_50_pollution_classes = ['hellicopter', 'chainsaw', 'siren', 'car_horn',
                                'engine', 'train', 'airplane']
    detected_pollution_classes = []
    for audio_class in list(counter.keys()):
        if audio_class in ESC_50_pollution_classes:
            audio_pollution_class = "Pollution source detected!"
            detected_pollution_classes.append(audio_class)
            pollution_source = detected_pollution_classes

    detected_classes_str = ', '.join(list(counter.keys()))
    detected_pollution_classes_str = ', '.join(detected_pollution_classes)
    audio_caption = ''
    if len(detected_pollution_classes) > 1:
        audio_caption = 'The following audio classes have been detected: ' + detected_classes_str + ', of which, ' + detected_pollution_classes_str + ' are considered to be related with air pollution events.'
    elif len(detected_pollution_classes) == 1:
        audio_caption = 'The following audio classes have been detected: ' + detected_classes_str + ', of which, ' + detected_pollution_classes_str + ' is considered to be related with air pollution events.'
    else:
        audio_caption = 'The following audio classes have been detected: ' + detected_classes_str + ', none of which is considered to be related with air pollution events.'

    response = {"audio_class": counter, "audio_pollution_class": audio_pollution_class, "caption": audio_caption,
                "pollution_source": pollution_source}
    print("Caller IP: " + request.remote_addr)
    print("Audio service: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " ", response)

    # delete file
    os.remove(filepath)
    return response

#audio_backend_api
@app.route('/api/audio/backend', methods=['GET', 'POST'])
def audio_backend():
    if request.method == 'POST':
        # prepare file
        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)

        audio_info_response = audio_info (filepath)
        audio_quality_response = audio_quality (filepath)

        response = {**audio_info_response, **audio_quality_response}


        # delete file
        os.remove(filepath)
        return response


# ***********************
# *** IMAGE ENDPOINTS ***
# ***********************

# image information
@app.route('/api/image/info', methods=['GET', 'POST'])
def image_info(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file if needed
        if filepath_alt == '':
            file = request.files['messageFile']
            # print(type(file))
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        # load image
        im = Image.open(filepath)
        if filepath_alt == '':
            os.remove(filepath)

        # image info
        format = im.format
        mode = im.mode
        # rgbimg = Image.new("RGB", im.size) # rgbimg.paste(im)
        width, height = im.size
        size = width*height/1000

        # response = f"""Image file format is {format}, type is {mode} and resolution is {size} MP."""
        # print(response)
        # respond
        response = {
            "imageformat": format,
            "imagetype": mode,
            "imageresolution": size
        }
        return response

# image quality
@app.route('/api/image/quality', methods=['GET', 'POST'])
def image_quality(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        if filepath_alt == '':
            file = request.files['messageFile']
            # print(type(file))
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        # load image
        im = Image.open(filepath)
        if filepath_alt == '':
            os.remove(filepath)

        # image quality
        im = im.convert('L')
        quality = brisque.score(im) / 10    # line 45 of `imquality/brisque.py’ -> if self.image.shape[-1] == 3:
        quality = int(round(quality))
        quality = min(max(quality, 1), 5)

        # image sharpness
        array = np.asarray(im, dtype=np.int32)
        gy, gx = np.gradient(array)
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        sharpness = int(round(sharpness))
        sharpness = min(max(sharpness, 1), 5)

        # image contrast
        stats = ImageStat.Stat(im)
        for band, name in enumerate(im.getbands()):
            contrast = (stats.stddev[band]-30)/6
            # print(f'Band: {name}, min/max: {stats.extrema[band]}, stddev: {stats.stddev[band]}')
            break
        contrast = int(round(contrast))
        contrast = min(max(contrast, 1), 5)

        # response = f"""Image sharpness is {sharpness}/5, contrast {contrast}/5 and quality is {quality}/5."""
        # print(response)
        # respond
        response = {
            "image sharpness": sharpness,
            "image contrast": contrast,
            "image quality": quality
        }
        return response

# image classification
@app.route('/api/image/classification', methods=['GET', 'POST'])
def image_classification(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file
        if filepath_alt == '':
            file = request.files['messageFile']
            # print(type(file))
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        # load file
        image = load_img(filepath, target_size=(224, 224))

        # remove file
        if filepath_alt == '':
            os.remove(filepath)

        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        yhat = model_image_classification.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]

        #beelexicon lookup
        # with open('beelexicon.txt', 'r') as file:
        #     # read all content of a file
        #     content = file.read()
        #     # check if string present in a file
        #     if label[1] in content:
        #         print (label[1])
        #         print('string exist in a file')
        #     else:
        #         print (label[1])
        #         print('string does not exist in a file')
        # print the classification
        # print('%s (%.2f%%)' % (label[1], label[2] * 100))
        # return f"""Image contains: {label[1]}."""
        # respond
        response = {
            "imageclass": label[1],
        }
        return response

# image captioning
@app.route('/api/image/captioning', methods=['GET', 'POST'])
def image_captioning(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file
        if filepath_alt == '':
            file = request.files['messageFile']
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        # load image (tou malaka)
        # features = feature_extractions(filepath, model_image_classification)

        # load image (ours)
        image = load_img(filepath, target_size=(224, 224))

        if filepath_alt == '':
            os.remove(filepath)

        # produce caption (tou malaka)
        # for i, filename in enumerate(features.keys()):
        #     vocab_size = tokenizer_image_captioning.num_words  # The number of vocabulary
        #     max_length = 37  # Maximum length of caption sequence
        #     label = sample_caption(model_image_captioning, tokenizer_image_captioning, max_length, vocab_size, features[filename])

        # produce catpion (ours)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model_image_classification.predict(image)
        t_label = decode_predictions(yhat)
        label = 'The following objects have been detected: '
        for item in t_label[0]:
            if item[2] > 0.1: label += item[1] + ', '
        label = label.replace("_", " ");
        label = label[:-2]

        # respond
        response = {
            "caption": label
        }
        # return f"""Image presents {label}."""
        return response

#image backend api
@app.route('/api/image/backend', methods=['GET', 'POST'])
def image_backend():
    if request.method == 'POST':
        # prepare file
        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)
        image_info_response = image_info (filepath)
        image_quality_response = image_quality (filepath)

        response = {**image_info_response, **image_quality_response}
        # delete file
        os.remove(filepath)
        return response

#image academe api
@app.route('/api/image/academe', methods=['GET', 'POST'])
def image_academe():
    if request.method == 'POST':
        # prepare file
        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)

        image_captioning_response = image_captioning(filepath)
        image_classification_response = image_classification (filepath)
        # print(image_classification_response)
        # print (image_classification_response['imageclass'])
        # beelexicon lookup

        with open('beelexicon.txt', 'r') as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            if image_classification_response['imageclass'] in content:
                # print(image_classification_response)
                # print('string exist in a file')
                pollution_response = {
                    "pollution_class": "polluting",
                }
            else:
                pollution_response = {
                    "pollution_class": "non-polluting",
                }

        response = {**image_captioning_response, **pollution_response}
        # response = image_captioning_response

        # delete file
        os.remove(filepath)
        return response

#Image-driven Bee-MATE
@app.route('/api/image/beemate', methods=['POST'])
def image_beemate(filepath_alt=''):
    if request.method == 'POST' or filepath_alt != '':
        # file = request.files['messageFile']
        # filepath = './temp/' + file.filename
        # file.save(filepath)
        #
        # image_classification_response = image_classification(filepath)

        file = request.files['messageFile']
        filepath = './temp/' + file.filename
        file.save(filepath)
        image_caption = image_captioning(filepath)

        # load image (ours)
        img = load_img(filepath, target_size=(224, 224))

        image = img_to_array(img)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model_image_classification.predict(image)
        t_label = decode_predictions(yhat)
        detected_classes = []
        for item in t_label[0]:
            if item[2] > 0.1: detected_classes.append(item[1])
        #print (detected_classes)
        #nikos edits
        detected_pollution_classes = []

        # classes = list(image_classification_response.values())

        # classes = [item for sublist in classes for item in sublist]
        # print('2')
        # print(classes)
        # classes = [x for x in classes if 's to ' not in x]
        # print('3')
        # print (classes)
        # counter = Counter(classes)

        # counter = {k: v / len(classes) for k, v in counter.items()}

        pollution_source = []
        # METHOD1 -  audio_pollution binary classification model
        # pollution = list(audio_pollution_response.values())
        # pollution = [item for sublist in pollution for item in sublist]
        # print (pollution)
        # if "others - polluting" in pollution:
        #     audiopollutionclass = "polluting"

        # METHOD2 - audio classfication model - dictionary
        # ESC_50_pollution_classes = ['hellicopter', 'chainsaw', 'siren', 'car_horn',
        #                             'engine', 'train', 'airplane']
        detected_pollution_classes = []


        with open('beelexicon.txt', 'r') as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            for image_class in detected_classes:
                print (image_class)
                if image_class in content:
                    audio_pollution_class = "Pollution source detected!"
                    detected_pollution_classes.append(image_class)
                    pollution_source = detected_pollution_classes
            # if image_classification_response['imageclass'] in content:
            #     image_pollution_source = {
            #         "pollution_source": image_classification_response['imageclass'],
            #     }
            # else:
            #     image_pollution_source = {
            #         "pollution_source": "none",
            #     }

        # with open('beelexicon.txt', 'r') as file:
        #     # read all content of a file
        #     content = file.read()
        #     # check if string present in a file
        #     if image_classification_response['imageclass'] in content:
        #         image_pollution_source = {
        #             "pollution_source": image_classification_response['imageclass'],
        #         }
        #     else:
        #         image_pollution_source = {
        #             "pollution_source": "none",
        #         }
        # response = {**image_caption, **image_pollution_source}
        if not pollution_source: pollution_source=["none"]
        response = {**image_caption, "pollution_source": pollution_source}
        print("Caller IP: " + request.remote_addr)
        print("Image service: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " ", response)

        # delete file
        os.remove(filepath)
        return response

#response = {"audio_class": counter, "audio_pollution_class": audio_pollution_class, "audio_caption": audio_caption,
#                                                                                                                                                                           "pollution_source": pollution_source}


# ***********************
# *** OTHER ENDPOINTS ***
# ***********************

# root point
@app.route('/api', methods=['GET'])
def api():
    # return render_template('home.html')
    return 'hello ΒΕΕ'


# *********************
# *** APP & SERVER  ***
# *********************

# start app and server
if __name__ == '__main__':

    print("loading models, please wait...")

    # load 1d_3_class_smo_cnn1d model
    model_3_class_smo_cnn1d = keras.models.load_model("./models/model_3_class_smo_cnn1d.h5")

    # load 1d_esc_50_cnn1d model
    model_cnn1d_esc_50 = keras.models.load_model("./models/model_cnn1d_esc_50.h5")

    # load 2_class_pollution_cnn1d model
    model_2_class_pollution_cnn1d = keras.models.load_model("./models/model_2_class_pollution_cnn1d.h5")

    # load image_classification model
    model_image_classification = VGG16()

    # load image_captioning model & tokenizer
    tokenizer_image_captioning = tokenizer_from_json(json.load(open('./models/tokenizer.json', 'r')))
    model_image_captioning = keras.models.load_model("./models/sample_model.h5")

    print("models loaded")

    print("now serving on http://m3capps.jour.auth.gr/api ...")
    serve(app, host='0.0.0.0', port=8001)
    app.run(debug=True)