
import json
import os

# with open('intents.json', 'r') as f:
#     intents = json.load(f)
#     intents_list = intents["intents"]


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


with open('intents.json', 'r') as jsonfile:
    intents = json.load(jsonfile)
    intents_list = intents["intents"]


for filename in os.listdir("./Intents [Dialogs]/"):
    if filename.startswith("_") and filename.endswith('.txt'):
        continue

    with open('./Intents [Dialogs]/' + filename, 'r') as f:

        patterns = []
        responses = []
        base = os.path.basename(f.name)
        tag = os.path.splitext(base)[0].lower()

        start_pattern = ''
        for line in nonblank_lines(f):

            # if tag.lower() == 'confirmation no':
            #     print(line.lower())

            if line[0] == '-' or line[0] == '\t':
                continue
            if line.lower() == tag.lower():
                start_pattern = tag.lower()
                continue
            if line == '<Response>':
                start_pattern = 'response'
                continue
            if start_pattern.lower() == tag.lower():
                patterns.append(line.lower())
                continue

            if start_pattern.lower() == 'response':
                responses.append(line.lower())
                continue

        # print(responses)
        intents["intents"].append({
            "tag": tag.lower(),
            "patterns": patterns,
            "responses": responses
        })

print(intents)
jsonstring = json.dumps(intents)
jsonfile = open('intents.json', 'w')
jsonfile.write(jsonstring)
jsonfile.close()
