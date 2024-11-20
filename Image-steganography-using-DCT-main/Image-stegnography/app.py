import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER


# DCT embedding and extraction functions
def embed_dct_message(input_image_path, text):
    # Load an image and convert it to a NumPy array
    image = np.array(Image.open(input_image_path))

    # Apply the 2D DCT to the image
    dct_image = np.fft.dct(np.fft.dct(image.T, norm="ortho").T, norm="ortho")

    # Embed data
    message_bytes = text.encode('utf-8')
    message_bits = ''.join(format(byte, '08b') for byte in message_bytes)

    data_index = 0
    for i in range(len(dct_image)):
        for j in range(len(dct_image[0])):
            if data_index < len(message_bits):
                dct_image[i][j] = (dct_image[i][j] // 2) * 2 + int(message_bits[data_index])
                data_index += 1

    # Invert the DCT to recover the image
    inverted_image = np.fft.idct(np.fft.idct(dct_image.T, norm="ortho").T, norm="ortho")

    # Save the modified image
    output_image = Image.fromarray(np.uint8(inverted_image))
    output_image.save(input_image_path)  # Overwrite the input image


def extract_dct_message(image_path):
    # Load the modified image and convert it to a NumPy array
    image = np.array(Image.open(image_path))

    # Apply the 2D DCT to the modified image
    dct_image = np.fft.dct(np.fft.dct(image.T, norm="ortho").T, norm="ortho")

    # Extract_data
    message_bits = ""
    data_index = 0
    for i in range(len(dct_image)):
        for j in range(len(dct_image[i])):
            if data_index < 8:  # Each character is 8 bits
                message_bits += str(dct_image[i][j] % 2)
                data_index += 1

    # Convert the extracted bits back to text
    extracted_message = "".join(
        chr(int(message_bits[i:i + 8], 2)) for i in range(0, len(message_bits), 8)
    )

    return extracted_message


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        text = request.form['text']

        if uploaded_file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filename)

            embed_dct_message(filename, text)
            return render_template("download.html", message="Data embedded successfully!", image=uploaded_file.filename)

    return render_template('home.html')


@app.route('/extract', methods=['POST'])
def extract():
    extracted_message = None

    if request.method == 'POST':
        uploaded_file = request.files.get('image')

        if uploaded_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            extracted_message = extract_dct_message(image_path)

    return render_template('extract.html', extracted_message=extracted_message)


@app.route('/download_embedded_image')
def download_embedded_image():
    embedded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embedded_image.png')
    return send_file(embedded_image_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
