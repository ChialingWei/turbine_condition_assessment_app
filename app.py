from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from util import *
from PIL import Image
from io import BytesIO
import numpy as np
import io
import matplotlib.image as mpimg
import image_similarity as imgsim

app = Flask(__name__)

width, w_crop, h_crop = 1120, 56, 56

@app.route('/')
def index():
    with open("index2.html") as file:
        return file.read()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global image_array
    image_file = request.files['image']
    image = Image.open(image_file)
    image_array = np.array(image)
    return jsonify({'message': 'Image uploaded successfully'}), 200

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        global points_data_global
        # Get the points data from the request
        points_data = request.json
        # Process the points data as needed
        print('Received points:', points_data)
        points_data_global = points_data
        # Return a response (optional)
        return jsonify({'message': 'Points received successfully'}), 200

@app.route('/get_crop_image', methods=['GET'])
def get_crop_image():
    crop_img_ori_size, _, _, _, _, _ = crop_img(image_array, points_data_global, 5)
    im = Image.fromarray(crop_img_ori_size)
    # im.save('crop_img.jpg')
    return send_file('crop_img.jpg', mimetype='image/jpeg')

@app.route('/trans', methods=['GET'])
def trans():
    global points_data_global
    global image_array
    _, cropped_img, crop_x, crop_y, zoom_crop_y, zoom_crop_x = crop_img(image_array, points_data_global, 5)
    tf_img = homography(cropped_img, points_data_global, crop_x, crop_y, zoom_crop_y, zoom_crop_x)  #numpy arr type
    img_path = 'transform_upload_img.png'
    # mpimg.imsave(img_path, tf_img)
    sequence_crop_image(w_crop, h_crop, img_path, 'crop_img')
    return send_file(img_path, mimetype='image/jpeg')

@app.route('/uploadTile', methods=['POST'])
def uploadTile():
    global tileNum
    if request.method == 'POST':
        tilePt = request.json
        print('Received points:', tilePt)
        tileNum = int(width/w_crop * math.floor(tilePt[1]/h_crop) + math.floor(tilePt[0]/w_crop) + 1)
        print('Tile Number', tileNum)
        # Return a response (optional)
        return jsonify({'message': 'Points received successfully'}), 200

@app.route('/simTile', methods=['GET'])
def simTile():
    global tileNum
    img_path = f"crop_img/{tileNum}.png"
    ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
    ImgSim.embed_dataset('crop_img/')
    num_lst, heatmap_d = ImgSim.similar_images(img_path, sim_thr=0.8, n=5)
    print(num_lst)
    # Return a response (optional)
    return jsonify({'heatmap_d': heatmap_d, 'message': 'Points received successfully'}), 200


app.run(debug=True)

