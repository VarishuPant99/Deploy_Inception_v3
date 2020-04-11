import sys,os,shutil
sys.path.append(os.getcwd())
from src.inference import  Inference
from flask import Flask, request, flash, redirect, render_template, url_for, Markup, send_file,send_from_directory
from werkzeug.utils import secure_filename

infer = Inference()
app = Flask(__name__)
app.secret_key = "!@#$%^&*()a-=afs;'';312$%^&*k-[;.sda,./][p;/'=-0989#$%^&0976678v$%^&*(fdsd21234266OJ^&UOKN4odsbd#$%^&*(sadg7(*&^%32b342gd']"
# the upload path for all the files

SUB_UPLOAD_FOLDER = "static/uploadFolder"
UPLOAD_FOLDER ="flask_api/"+ SUB_UPLOAD_FOLDER
# a list to track all the files loaded in memory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
list_of_uploaded_file = []


def clean_upload_folder():
    try:
        shutil.rmtree(UPLOAD_FOLDER + "/")
    except FileNotFoundError as e:
        pass


def make_directory():
    os.makedirs(UPLOAD_FOLDER,exist_ok=True)


@app.route('/')
def root():
    # clean the upload directory every time user use the website and create a new empty directory
    clean_upload_folder()
    make_directory()
    return render_template("upload.html")

@app.route("/upload",methods=["POST","GET"])
def upload_image_file():
    print("root testing" , request.files)
    if request.method == "POST":
        # check wether the request value is 
        print("upload file" , request.files)
        if "file" in request.files:
            # get the multiDict
            file = request.files.getlist("file")[0]
            print("upload file" , request.files)
            # secure the filename which will only give file name excluding other parameters
            filename = secure_filename(file.filename)
            #print(filename , "  " , dir(file.stream))
            # get the file path
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            result  = infer(path)
            return render_template("result.html",result=result,image_path = os.path.join(SUB_UPLOAD_FOLDER,filename))


@app.route("/test")
def test():
    return "Hello World"



if __name__  =="__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
