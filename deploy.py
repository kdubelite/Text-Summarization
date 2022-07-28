from flask import Flask, request, jsonify, render_template
from getData import get_transcript

from model import *

# create the flask app
app = Flask(__name__)

# what html should be loaded as the home page when the app loads?
@app.route('/')
def home():
    return render_template('page.html', prediction_text="")

# define the logic for reading the inputs from the WEB PAGE, 
# running the model, and displaying the prediction
@app.route('/predict', methods=['GET','POST'])
def predict():

    # get the description submitted on the web page
    a_description = request.form.get('description')
    outData = get_transcript(a_description)

    with open(r'out.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in outData))

    with open('out.txt', 'r') as file:
        mdData = file.read().replace('\n', '')    

    predData = learn.blurr_generate(mdData, early_stopping=False, num_return_sequences=1)
    outSummary = predData[0]['generated_texts']
    return render_template('page.html', prediction_text=outSummary)
    #return 'Description entered: {}'.format(a_description)


# boilerplate flask app code
if __name__ == "__main__":
    app.run(debug=True)
