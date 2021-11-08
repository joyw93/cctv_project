import json


with open('C:\projects\cctvproject\yolov5\data.json') as f:
    data = json.load(f)

print(data[0]['cctvname'])