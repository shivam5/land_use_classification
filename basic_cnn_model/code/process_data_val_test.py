import os
import json
import cv2
from random import shuffle

val_data_path = "../Data/compressed_val/"

save_val_data_path = "../Data/processed_val/images/"
val_label_file = "../Data/processed_val/labels.txt"

save_test_data_path = "../Data/processed_test/images/"
test_label_file = "../Data/processed_test/labels.txt"

val_categories_folders = os.listdir(val_data_path)

val_label_file = open(val_label_file, "w") 
test_label_file = open(test_label_file, "w") 

labels = {
"crop_field":1,
"military_facility":2,
"educational_institution":3,
"place_of_worship":4,
"recreational_facility":5,
"solar_farm":6

}

for one_category_folder in val_categories_folders:
    data_path = os.path.join(val_data_path, one_category_folder)
    files = sorted(os.listdir(data_path))
    # print(len(files))
    shuffle(files)
    # print(len(files))
    val_files = sorted(files[0:110])
    test_files = sorted(files[110:220])

    for file in val_files:
        path = os.path.join(data_path, file)
        json_name = file + '_0_msrgb.json'
        json_file = os.path.join(path, json_name)
        json_data=open(json_file)
        data = json.load(json_data)
        
        bbox = data['bounding_boxes'][0]['box']

        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x1 = int(x)
        x2 = int(x+w)
        y1 = int(y)
        y2 = int(y+h)

        img_name = data['img_filename']
        img_path = os.path.join(path, img_name)
        save_img_path = os.path.join(save_val_data_path, img_name)
        save_img_path = save_img_path.replace('_0_msrgb','')

        img = cv2.imread(img_path)
        cropped_img = img[y1:y2, x1:x2, :]
        cv2.imwrite(save_img_path, cropped_img)

        val_label_file.write(img_name.replace('_0_msrgb',''))
        val_label_file.write(":")
        val_label_file.write(str(labels[one_category_folder]))
        val_label_file.write("\n")

        json_data.close()

    for file in test_files:
        path = os.path.join(data_path, file)
        json_name = file + '_0_msrgb.json'
        json_file = os.path.join(path, json_name)
        json_data=open(json_file)
        data = json.load(json_data)
        
        bbox = data['bounding_boxes'][0]['box']

        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x1 = int(x)
        x2 = int(x+w)
        y1 = int(y)
        y2 = int(y+h)

        img_name = data['img_filename']
        img_path = os.path.join(path, img_name)
        save_img_path = os.path.join(save_test_data_path, img_name)
        save_img_path = save_img_path.replace('_0_msrgb','')

        img = cv2.imread(img_path)
        cropped_img = img[y1:y2, x1:x2, :]
        cv2.imwrite(save_img_path, cropped_img)

        test_label_file.write(img_name.replace('_0_msrgb',''))
        test_label_file.write(":")
        test_label_file.write(str(labels[one_category_folder]))
        test_label_file.write("\n")

        json_data.close()

