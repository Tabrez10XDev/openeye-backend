from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from detect import diagnose, features
import base64
import io
from PIL import Image
app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
# parser.add_argument('image', type=Image, help='Base64-encoded image data', required=True)


class Features(Resource):

    def get(self):
        return jsonify({'features': "features()"})


class Detect(Resource):

    def get(self):
    
        image = Image.open("imgs.jpg")
        disease = diagnose(image)
        return jsonify({
            "result": disease
        })

    def post(self):
        
        try:
            uploaded_file = request.files['image']
            if uploaded_file:
                image = Image.open(uploaded_file)
                disease = diagnose(image)

            return jsonify({"result": disease})
        except Exception as e:
            print(e)
            return jsonify({"error": "Invalid image data or error processing the image"})


api.add_resource(Features, '/')
api.add_resource(Detect, '/detect')

# driver function
app.run(debug=True)