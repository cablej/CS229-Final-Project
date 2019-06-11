# Helper script to convert json from hydrator to only tweets

import json

filename = "data/tweets-2016-2-100000"

with open(filename + "-full.txt") as input_file:
	with open(filename + "-textonly.txt", 'w') as write:
		for line in input_file:
			# print(line)
			tweet = json.loads(line)
			text = tweet['full_text']
			text = text.replace('\n', ' ')
			text = text.replace('\r', ' ')
			if len(text) < 2:
				continue
			write.write(tweet['full_text'] + '\n')
