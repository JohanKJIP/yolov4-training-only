import os
import shutil
import cv2


def collect_and_rename() -> None:
    """ Collect images and labels and place them into a single directory """
    image_source_folder = 'image_dir'
    label_source_folder = 'annotation_dir'
    image_target_folder = 'images'
    label_target_folder = 'labels'
    count = -1
    for subdir, _, files in os.walk(image_source_folder):
        # it walks the parent folder first
        if count == -1: 
            count += 1
            continue
        subdir_name = subdir.split('\\')[1]
        for file_name in files:
            with open(f'{image_source_folder}/{subdir_name}/{file_name}') as image_file, \
                open(f'{label_source_folder}/{subdir_name}/{file_name}'.split('.')[0] + '.txt') as label_file:
                shutil.copy2(image_file.name, f'{image_target_folder}/{"%06d" % count}.jpg')
                shutil.copy2(label_file.name, f'{label_target_folder}/{"%06d" % count}.txt')
            count += 1
            print(f'Processed {count} images')


def convert_labels() -> None:
    """ Convert labels from YOLO format to COCO format and place in train/val.txt """
    data_folder = 'images'
    validation_split = 0.10

    # Convert annotations and split into validation and train set
    number_images = len(os.listdir(data_folder)) / 2
    train_size = int(number_images * (1 - validation_split))
    val_size = number_images - train_size

    print(f'Training dataset size: {train_size}')
    print(f'Validation dataset size: {val_size}')

    count = 0
    with open('train.txt', 'w') as train_file, open('val.txt', 'w') as val_file:
        for file_name in os.listdir(data_folder):
            if file_name.split('.')[1] == 'jpg':
                write = False
                if count < train_size:
                    file_to_write = train_file
                else:
                    file_to_write = val_file

                with open(f'{data_folder}/{file_name}'.split('.')[0] + '.txt') as label_file:
                    labels = []
                    for line in label_file:
                        line = line.split(' ')
                        line[-1] = line[-1].rstrip()

                        img = cv2.imread(f'{data_folder}/{file_name}')
                        img_height = img.shape[0]
                        img_width = img.shape[1]
                        
                        x = float(line[1]) * img_width
                        y = float(line[2]) * img_height
                        w = float(line[3]) * img_width
                        h = float(line[4]) * img_height

                        xmin = int(x - w/2)
                        ymin = int(y - h/2)
                        xmax = int(x + w/2)
                        ymax = int(y + h/2)

                        labels.append(f' {xmin},{ymin},{xmax},{ymax},{line[0]}')
                    if len(labels) > 0:
                        write = True
                        file_to_write.write(f'{data_folder}/{file_name}')
                        for label in labels:
                            file_to_write.write(label)
                if write:
                    file_to_write.write('\n') 
                print(f'[{count}/{number_images}] Processed {file_name}')
                count += 1

if __name__ == "__main__":
    convert_labels()