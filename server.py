from flask import Flask, flash, redirect, render_template, request, session, abort
from run_all import nb_predict

print('Classifier trained.')

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		tweet = request.form['tweet']
		pred, prob = nb_predict(tweet)
		print(pred[0])
		print(prob[0])
		return render_template('template.html', tweet=tweet, pred=pred[0], prob=prob[0])
	else:
		return render_template('template.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)