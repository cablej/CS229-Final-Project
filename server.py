from flask import Flask, flash, redirect, render_template, request, session, abort
from run_all import nb_predict
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from flask import jsonify
import threading

#import OAuth secrets from secret.py
import secrets

print('Classifier trained.')

last_tweet = None

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

@app.route("/live", methods=['GET'])
def live():
	return render_template('live.html')

@app.route("/livedata", methods=['GET'])
def livedata():
	global last_tweet
	pred, prob = nb_predict(last_tweet.text)
	return jsonify({
		'prediction': int(pred[0]),
		'probability': float(prob[0][1]),
		'text': last_tweet.text,
		'author': last_tweet.author.screen_name
		})


class StdOutListener(StreamListener):
	""" A listener handles tweets that are received from the stream.
	This is a basic listener that just prints received tweets to stdout.
	"""
	def on_status(self, status):
		global last_tweet
		if(status.text.startswith('RT')):
			return
		print(status.text)
		last_tweet = status
		return True

	def on_error(self, status_code):
		if status_code == 420:
			return False

def ingest():

	l = StdOutListener()
	auth = OAuthHandler(secrets.consumer_key, secrets.consumer_secret)
	auth.set_access_token(secrets.access_token, secrets.access_token_secret)

	stream = Stream(auth, l)
	stream.filter(track=['trump'])

if __name__ == "__main__":

	thread = threading.Thread(target = ingest)
	thread.start()

	app.run(host='0.0.0.0', port=80)
