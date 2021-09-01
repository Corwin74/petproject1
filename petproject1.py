from flask import Flask
from flask import render_template, request
import numpy as np
from PIL import Image
import base64
import re
import sys
from io import StringIO
import logging
logging.basicConfig(filename='record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("hello.html")

@app.route('/hook', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    app.logger.info(image_b64)
#    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
#    image_PIL = Image.open(StringIO(image_b64))
#    image_np = np.array(image_PIL)
#    print('Image received: {}'.format(image_np.shape))
    return ''

if __name__ == "__main__":
    app.run(host='0.0.0.0')