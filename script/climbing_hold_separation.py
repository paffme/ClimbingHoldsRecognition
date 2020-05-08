import json

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

        annotation_id = 0
        new_annotations = []

        for annotations in coco_file['annotations']:
            tmp = annotations.copy()
            tmp['segmentation'] = []
            tmp['id'] = annotation_id
            for cnt in range(0, len(annotations['segmentation'])):
                tmp['segmentation'].append(annotations['segmentation'][cnt])
                tmp['id'] = annotation_id
                new_annotations.append(tmp.copy())

                tmp['segmentation'] = []
                annotation_id += 1

        coco_file['annotations'] = new_annotations
        with open(args.coco_file, 'w') as outfile:
            outfile.write(json.dumps(coco_file, indent=4))
