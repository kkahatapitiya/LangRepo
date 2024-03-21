import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', type=json.loads)
args = parser.parse_args()
print(type(args.list[0]))
