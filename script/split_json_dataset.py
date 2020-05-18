import json
from random import randint
import os

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Split coco dataset with train conf and val conf.')
    parser.add_argument("--coco-file",
                        metavar="/path/and/file/name",
                        help="Enter valid coco file with train and val as subdir")
    args = parser.parse_args()

    with open(args.coco_file, "r") as read_file:
        coco_file = json.load(read_file)

        coco_train_files = []
        train_img_id = []
        coco_val_files = []
        val_img_id = []

        path_dataset = os.path.abspath(args.coco_file[:str(args.coco_file).rindex('/') + 1])

        nb_images = len(coco_file['images'])
        nb_images_train = nb_images * 0.8

        # Split with ratio : {train: 80%, val: 20%}
        for img in coco_file['images']:

            img['events'] = []

            if (randint(0, nb_images - 1) < nb_images_train and len(coco_train_files) < nb_images_train) \
                    or len(coco_val_files) >= (nb_images * 0.2):
                coco_train_files.append(img)
                train_img_id.append(img["id"])
                os.symlink(path_dataset + "/" + img["file_name"], path_dataset + "/train/" + img["file_name"])
            else:
                coco_val_files.append(img)
                val_img_id.append(img["id"])
                os.symlink(path_dataset + "/" + img["file_name"], path_dataset + "/val/" + img["file_name"])

        coco_train_anns = []
        coco_val_anns = []

        for ann in coco_file['annotations']:
            ann['events'] = []
            if ann['image_id'] in train_img_id:
                coco_train_anns.append(ann)
            elif ann['image_id'] in val_img_id:
                coco_val_anns.append(ann)

        # print(coco_train_files)

        coco_train = json.dumps({'images': coco_train_files,
                                 'categories': coco_file['categories'],
                                 'annotations': coco_train_anns},
                                indent=4
                                )
        coco_val = json.dumps({'images': coco_val_files,
                               'categories': coco_file['categories'],
                               'annotations': coco_val_anns},
                              indent=4
                              )

        indexStart = str(args.coco_file).rindex("/")
        indexEnd = str(args.coco_file).rindex(".json")

        path = args.coco_file[0: indexStart]

        with open(path + "/train" + args.coco_file[indexStart: indexEnd] + "_train.json", 'w') as outfile:
            outfile.write(coco_train)
        with open(path + "/val" + args.coco_file[indexStart: indexEnd] + "_val.json", 'w') as outfile:
            outfile.write(coco_val)
