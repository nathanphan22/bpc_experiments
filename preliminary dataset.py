import os
import json
from PIL import Image

# Folder containing the .txt files
folder_path = 'custom_images_contours/polygons'

# Dictionary to store file contents
data = {}
os.makedirs('annotations', exist_ok=True)

# Loop through files in the folder
for foldername in os.listdir(folder_path):
    new_path = folder_path + "/" + foldername
    for filename in os.listdir(new_path):

        if filename.endswith('.txt'):
            file_path = os.path.join(new_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data['image_path'] = 'images/' + foldername + '-' + filename.strip('.txt') + '.png'
                data['height'] = 480
                data['width'] = 640
                data['masks'] = []

                polygon = {}
                polygon['label'] = 'obj_' + foldername
                polygon['points'] = []
                for line in file:
                    if line == '\n' :
                        break
                    x = int(line.split(', ')[0])
                    y = int(line.split(', ')[1])
                    polygon['points'].append([x,y])
                data['masks'].append(polygon)
                

                # Save the dictionary to a .json file
                output_path = 'annotations/' + foldername + '-' + filename.strip('.txt') + '.json'
                with open(output_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)

                print(f"JSON file saved to {output_path}")

image_folder_path = 'custom_images'
os.makedirs('images', exist_ok=True)

for foldername in os.listdir(image_folder_path):
    new_path = image_folder_path + "/" + foldername
    for filename in os.listdir(new_path):
        if filename.endswith('.png'):
            file_path = os.path.join(new_path, filename)

            img = Image.open(file_path)
            output_path = 'images/' + foldername + '-' + filename
            img.save('images/' + foldername + '-' + filename)

            print(f"png file saved to {output_path}")