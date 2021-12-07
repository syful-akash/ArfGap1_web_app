"""Custome/written script"""
from ObjectDetector import Detector #self written script
import img_transforms #written script 

"""Standard libraries"""
import io
from flask import Flask, render_template, request, send_from_directory, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import os
import cv2

''' __name__ is a convinient shorcut which helps Flsk to find 
where the necessary resources (Templates and statics) are'''
app = Flask(__name__)  
 


'''following calls the Detector() class from ObjectDetector script. 
This Detector() class is actually our model that we train to detect 
Nucleus and Golgi. This Class return the Detected version of given image.'''
detector = Detector()

'''What is RENDER_FACTOR?'''
RENDER_FACTOR = 5

'''------------------Image Save Destination Folder------------'''
#destination folder where we will save all uploaded and detected images
uploads_dir ='./static/'
os.makedirs(uploads_dir, exist_ok=True)

''''----------Necessary Functions for inference---------'''
# function to load img from url. This is an extra Feature.
def load_image_url(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img

#def run_inference_transform(img_path = 'file.jpg', transformed_path = 'file_transformed.jpg'):
	'''Extra feature of this app. it can make the inference process fast
	by reducing input image size'''

	# get height, width of image
	original_img = Image.open(img_path)

	# transform to square, using render factor
	transformed_img = img_transforms._scale_to_square(original_img, targ=RENDER_FACTOR*16)
	transformed_img.save(transformed_path)

	# run inference using detectron2
	untransformed_result = detector.inference(transformed_path)

	# unsquare
	result_img = img_transforms._unsquare(untransformed_result, original_img)

	# clean up
	try:
		os.remove(img_path)
		os.remove(transformed_path)
	except:
		pass

	return result_img

'''This Function is responsible for detection.'''
#def run_inference(img_path = 'file.jpg'): #why 'file.jpg' assigned into the img_path argument?
#
#	# run inference using detectron2
#	result_img = detector.inference(img_path)
#	img_path = os.path.join(dest_dir,img_path)
#	result_img.save(img_path)
#
#	#clean up
#	#try:
#	#	os.remove(img_path)
#	#except:
#	#	pass
#
#	return result_img
#	#return img_path

'''-------------------Rounting and Rendering html templates--------'''

#@app.route("/detect", methods=['POST', 'GET'])
sayem = 5
@app.route("/", methods=['POST', 'GET'])
def upload():
	if request.method == 'POST':
		try:

			# upload image
			#requesting nucleusimage file to upload
			nucleus = request.files['file']
			#requesting golgi image file to upload
			golgi = request.files['gfile']

			#filename
			nfilename = secure_filename(nucleus.filename) #nucleus
			gfilename = secure_filename(golgi.filename) #golgi
			print(nfilename) #if you want to see the file name in the terminal uncmnt this line
			print(gfilename)
			#file path
			npath = os.path.join(uploads_dir, nfilename) # nucleus file directory
			gpath = os.path.join(uploads_dir, gfilename) # golgi file directory

			#saving uploaded image 
			nucleus.save(npath) #saving uploaded nucleus image
			golgi.save(gpath) #saving uploaded golgi image
			print(npath)
			print(gpath)
			#following open the uploaded image
			nfile = Image.open(npath) #opeing nucleus
			gfile = Image.open(gpath) #opeing nucleus

			# remove alpha channel
			nrgb_im = nfile.convert('RGB') #nucleus
			grgb_im = gfile.convert('RGB') #golgi
			#print('here is the data type',type(rgb_im)) #this is for troubleshoot
			#rgb_im.close()
			
			#saving rgb version of nucleus file to static folder
			nrgb_im.save(npath) #nucleus
			grgb_im.save(gpath) #golgi


		except:
			'''This render the failure.html if the image file is not 
			uploaded properly'''
			return render_template("failure.html")
			
	elif request.method == 'GET':
		'''in the following block we are Doing same thing as done with POST 
		method and file but this time with GET method and url  '''
		# get url
		url = request.args.get("url")

		# save
		try:
			# save image as jpg
			# urllib.request.urlretrieve(url, 'file.jpg')
			rgb_im = load_image_url(url)
			rgb_im = rgb_im.convert('RGB')

			#save image
			rgb_im.save('file.jpg')

		# failure
		except:
			#return render_template("failure.html")
			return render_template("index.html", image_loc=None)
			
	
	#following run the inference to the loaded nucleus image
	nresult_img, n_num = detector.inference(npath) #nucleus
	gresult_img, g_num = detector.inference(gpath) #golgi
	
	print(n_num, g_num)
	#following save the detected nucleus image
	nresult_img.save(npath) #nucleus
	gresult_img.save(gpath) #golgi

	# create file-object in memory
	nfile_object = io.BytesIO() #nucleus
	gfile_object = io.BytesIO() #golgi

	# write PNG in file-object
	nresult_img.save(nfile_object, 'PNG') #nucleus
	gresult_img.save(gfile_object, 'PNG') #golgi

	# move to beginning of file so `send_file()` it will read from start    
	nfile_object.seek(0) #nucleus
	gfile_object.seek(0) #golgi

	ratio = int(g_num)/int(n_num)
	if ratio < 0.9:
		result = 'Toxic'
	elif ratio > 0.9 and ratio <=1:
		result = 'Non-toxic'
	else: result = 'algorithm predicts golgy incorrectly' 


	#rendering index.html with the value of image_loc for the html script
	return render_template("index.html", image_loc=npath, image_loc1=gpath, n_num=n_num, g_num=g_num, ratio=ratio, result=result)
	
if __name__ == "__main__":

	# get port. Default to 8080
	port = int(os.environ.get('PORT', 8080))

	# run app
	app.run(host='0.0.0.0', port=port, debug=True)


