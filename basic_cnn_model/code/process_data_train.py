import os
import json
import cv2

train_data_path = "../Data/compressed_train/"
save_train_data_path = "../Data/processed_train/images/"
label_file = "../Data/processed_train/labels.txt"
train_categories_folders = os.listdir(train_data_path)

label_file = open(label_file, "w") 

labels = {
"crop_field":1,
"military_facility":2,
"educational_institution":3,
"place_of_worship":4,
"recreational_facility":5,
"solar_farm":6

}

for one_category_folder in train_categories_folders:
    data_path = os.path.join(train_data_path, one_category_folder)
    files = sorted(os.listdir(data_path))
    files = files[0:1700]

    for file in files:
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
        save_img_path = os.path.join(save_train_data_path, img_name)
        save_img_path = save_img_path.replace('_0_msrgb','')

        img = cv2.imread(img_path)
        cropped_img = img[y1:y2, x1:x2, :]
        cv2.imwrite(save_img_path, cropped_img)

        label_file.write(img_name.replace('_0_msrgb',''))
        label_file.write(":")
        label_file.write(str(labels[one_category_folder]))
        label_file.write("\n")

        json_data.close()

