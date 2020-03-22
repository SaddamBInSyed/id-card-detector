import unittest


class Tests(unittest.TestCase):

    def test_read_and_validate_coco_annotation(self):
        from id_card_detector.utils import read_and_validate_coco_annotation

        false_sample_list = ["tests/data/coco_false_"+str(ind)+".json" for ind in range(17)]
        true_sample_list = ["tests/data/coco_true_"+str(ind)+".json" for ind in range(2)]

        for false_sample in false_sample_list:
            _, response = read_and_validate_coco_annotation(false_sample)
            self.assertEqual(response, False)

        for true_sample in true_sample_list:
            _, response = read_and_validate_coco_annotation(true_sample)
            self.assertEqual(response, True)

    def test_process_coco(self):
        import jsonschema
        from id_card_detector.utils import process_coco

        # form json schemas
        segmentation_schema = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "number",
                    },
                "additionalItems": False
                },
            "additionalItems": False
        }

        annotation_schema = {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer"
                    },
                "category_id": {
                    "type": "integer"
                    },
                "segmentation": segmentation_schema
            },
            "required": ["image_id", "category_id", "segmentation"]
        }

        image_schema = {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string"
                    },
                "id": {
                    "type": "integer"
                    },
                "annotations": {
                    "type": "array",
                    "items": annotation_schema,
                    "additionalItems": False
                    }
            },
            "required": ["file_name", "id"]
        }

        image_list_schema = {
                "type": "array",
                "items": image_schema,
                "additionalItems": False
                }

        category_schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                    },
                "id": {
                    "type": "integer"
                    }
            },
            "required": ["name", "id"]
        }

        category_list_schema = {
                "type": "array",
                "items": category_schema,
                "additionalItems": False
                }

        # process sample coco file
        COCO_PATH = "tests/data/coco_true_0.json"
        images, categories = process_coco(COCO_PATH)

        # check if returned list lenghts are valid
        self.assertEqual(len(images), 2)
        self.assertEqual(len(categories), 2)

        # check if returned image fileds are valid
        self.assertEqual(images[1]["id"], 2)
        self.assertEqual(images[1]["file_name"], "data/midv500/images/example2.tif")
        self.assertEqual(images[1]["annotations"][0]["image_id"], 2)
        self.assertEqual(images[0]["annotations"][1]["image_id"], 1)
        self.assertEqual(images[0]["annotations"][1]["category_id"], 2)

        # check if returned images schema is valid
        try:
            jsonschema.validate(images, image_list_schema)
            validation = True
        except jsonschema.exceptions.ValidationError as e:
            print("well-formed but invalid JSON:", e)
            validation = False
        self.assertEqual(validation, True)

        # check if returned categories schema is valid
        try:
            jsonschema.validate(categories, category_list_schema)
            validation = True
        except jsonschema.exceptions.ValidationError as e:
            print("well-formed but invalid JSON:", e)
            validation = False
        self.assertEqual(validation, True)


if __name__ == '__main__':
    unittest.main()
