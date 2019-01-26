import json
import numpy as np
from PIL import Image, ImageDraw

def json_to_array(x):
    return np.array(json.loads(x))

def vector_to_pixel(vector_image, image_size=256):
    """
    Converts image in the original vector format into pixel format
    """
    image = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(image)
    for stroke in vector_image:
        for i in range(len(stroke[0]) - 1):
            draw.line((stroke[0][i], stroke[1][i]) + (stroke[0][i+1], stroke[1][i+1]), fill=255, width=6)
            draw.ellipse([stroke[0][i]-2, stroke[1][i]-2,
                          stroke[0][i]+2, stroke[1][i]+2], fill=255)
        draw.ellipse([stroke[0][len(stroke[0])-1]-2, stroke[1][len(stroke[0])-1]-2,
                      stroke[0][len(stroke[0])-1]+2, stroke[1][len(stroke[0])-1]+2], fill=255)
    return np.array(image.resize((image_size, image_size)))

def image_generator(dataset, include_y):
    batch_number = 0
    while True:
        if batch_number * batch_size > len(dataset):
            batch_number = 0
            dataset = dataset.sample(frac=1)
        first_index = batch_number * batch_size
        last_index = min((batch_number + 1) * batch_size, len(dataset))
        number_of_items = last_index - first_index
        X = np.zeros((number_of_items, image_size, image_size, 1))
        for i in range(first_index, last_index):
            X[i-first_index, :, :, 0] = vector_to_pixel(dataset['drawing'].iloc[i], image_size)
        X /= 255
        if include_y:
            y = dataset['word'].iloc[first_index:last_index].values.reshape(-1, 1)
            yield X, oneHotEncoder.transform(labelEncoder_y.transform(y).reshape(-1, 1)).toarray()
        else:
            yield X
        batch_number += 1