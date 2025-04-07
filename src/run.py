import classification_model

model = classification_model.ClassificationModel()
model.pumpen()

model.predict("/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen_big/beech/bark_4068_box_00_angle_-5.34.jpg")
model.predict("/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen_big/chestnut/bark_2628_box_00_angle_-3.95.jpg")
model.predict("/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen_big/pine/bark_4308_box_01_angle_9.19.jpg")