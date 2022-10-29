import json
import os

final_labels = {"info": {"description": "my-project-name"}, "images": [],
                "annotations": [], "categories": [{"id": 1, "name": "răng sâu"}]}


def addLabel(label):
    img_id = len(final_labels['images'])
    annotate_id = len(final_labels['annotations'])
    for i in label['images']:
        annotations = list(
            filter(lambda x: x['image_id'] == i['id'], label['annotations']))
        i['id'] = img_id+1
        final_labels['images'].append(i)
        for a in annotations:
            a['id'] = annotate_id
            annotate_id += 1
            a['image_id'] = img_id+1
            final_labels['annotations'].append(a)
        img_id += 1


for json_name in os.listdir("./label"):
    with open("./label/"+json_name, 'r') as f:
        label = json.load(f)
        addLabel(label)

image_in_annotation = list(map(lambda x: x["image_id"], label['annotations']))

images = list(filter(lambda x: x['file_name']
              in os.listdir('data') and x['id'] in image_in_annotation, final_labels['images']))


def func(x):
    x['category_id'] = 1
    return x


images = list(map(func, images))
final_labels['images'] = images

with open('label.json', 'w') as f:
    json.dump(final_labels, f)
