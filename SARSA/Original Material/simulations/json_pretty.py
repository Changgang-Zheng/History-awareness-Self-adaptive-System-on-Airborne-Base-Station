import json
import sys

file_name = sys.argv[1]
with open(file_name, 'r') as fil:
    parsed = json.load(fil)

with open(file_name, 'w') as fil:
    fil.write(json.dumps(parsed, indent=4, sort_keys=False))
