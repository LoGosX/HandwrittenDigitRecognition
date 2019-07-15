TODO: better readme

# What is it
Simple webpage where you can draw a digit using mouse and my model built with Tensorflow will predict what digit it is.  
The model is just a plain Neural Network with two hidden layers (100 neurons each, using ReLU activation).  
Built using Python 3.  

## How to use:
1. Clone this repository
2. Install required modules listed in requirements.txt (`pip install -r requirements.txt`)
3. Change directory to src. (`cd src`)
4. Type in bash/shell `export FLASK_APP=server.py`
5. start server with `flask run`
6. Open browser and go to http://localhost:5000/
7. Have fun drawing digits!

You can also play with it there https://afternoon-ridge-57848.herokuapp.com  
It may be need some time to load for the first time but don't worry, just wait.  
Sometimes it may not update predictions (slow server), then just click "predict" button on the bottom left.  
THIS SITE DOES NOT WORK ON MOBILE!  