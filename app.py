from flask import Flask
import requests
import pandas
from flask import Flask, redirect, url_for, render_template, request, flash
import app
import os
import skimage
from skimage import filters
from flask import Flask
import numpy as np
import cv2 as cv
import os
import pickle
import sys
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = '/static'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def home():
   response=False
   grey='not entered'
   return render_template('index.html')

@app.route('/transform', methods = ['GET', 'POST'])
def transform():
   if request.method == 'POST':
      
      f = request.files['uploaded_image']
      path=os.path.join(app.config['UPLOAD_FOLDER'], 'input')
      os.remove(path) 
      os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'))
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input'))
      selected=request.form.get('transformation')

      if(selected=='grayscale'):
         # image_numpy = pickle.loads(f.read())
         img = cv.imread(cv.samples.findFile(path),0)
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),img)
         return render_template('index.html',grey='grayscale',response=True)

      elif(selected=='denoising'):
         img = cv.imread(cv.samples.findFile(path),0)
         img_denoised=filters.median(img, selem=np.ones((5, 5)))
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),img_denoised)
         return render_template('index.html',grey='denoised',response=True)
      
      elif(selected=='edgedetector'):
         img = cv.imread(cv.samples.findFile(path),0)
         # edges = skimage.feature.canny(img, sigma=0) 
         edges = cv.Canny(img,100,200)
         # edges=edges.convert("RGB")
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),edges)
         return render_template('index.html',grey='Canny Edge Detector',response=True)

      elif(selected=='rotation'):
         img = cv.imread(cv.samples.findFile(path))
         num_rows, num_cols = img.shape[:2]
         img_rotation = cv.warpAffine(img, cv.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.6), (num_cols, num_rows))
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),img_rotation)
         return render_template('index.html',grey='image_rotation',response=True)
         
      elif(selected=='mirror'):
         img = cv.imread(cv.samples.findFile(path))
         num_rows, num_cols = img.shape[:2]
         src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1]])
         dst_points = np.float32([[num_cols-1,0], [0,0], [num_cols-1,num_rows-1]])
         matrix = cv.getAffineTransform(src_points, dst_points)
         img_afftran = cv.warpAffine(img, matrix, (num_cols,num_rows))
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),img_afftran)
         return render_template('index.html',grey='Mirror image',response=True)
         

      elif(selected=='morphology'):
         from skimage import color
         from skimage import morphology, segmentation   #Morphlogical transformations
         from skimage import measure
         from scipy.ndimage import distance_transform_edt
         img = cv.imread(cv.samples.findFile(path))
         dt = distance_transform_edt(img)
         
         labels = morphology.watershed(-dt)
         img_color=color.label2rgb(labels, image=img)
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),img_color)
         return render_template('index.html',grey='Morphological Color for image',response=True)

      elif(selected=='blur'):
         img = cv.imread(cv.samples.findFile(path))
         kernel = np.ones((5,5),np.float32)/25
         dst = cv.filter2D(img,-1,kernel)
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),dst)
         return render_template('index.html',grey='Sharpening / Blurring Image',response=True)

      elif(selected=='perspective'):
         img = cv.imread(cv.samples.findFile(path))
         rows,cols,ch = img.shape
         pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
         pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
         M = cv.getPerspectiveTransform(pts1,pts2)
         dst = cv.warpPerspective(img,M,(300,300))
         cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'),dst)
         return render_template('index.html',grey='Perspective Change',response=True)

      f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'output'))
      return render_template('index.html',grey="Not entered",response=False,note='Image is not uploaded / or error at code side')


if __name__ == '__main__':
    
    app.run(debug=True)
