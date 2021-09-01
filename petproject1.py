from flask import Flask, render_template, request
import base64
import re

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("hello.html")

@app.route('/hook', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)
    imgdata = base64.urlsafe_b64decode(image_data)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    return ''

if __name__ == "__main__":
    app.run(host='0.0.0.0')