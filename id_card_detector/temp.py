import id_card_detector
image_path="tests/data/idcard2.jpg"
model_name="maskrcnn_resnet50"
id_card_detector.detect_card(image_path="tests/data/idcard2.jpg", output_dir="outputs")