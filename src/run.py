import classification_model

model = classification_model.ClassificationModel()
model.pumpen()

model.predict()
model.predict("/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen/chestnut/bark_14604_box_00_angle_-4.67.jpg")
model.predict("/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen/pine/bark_7236_box_00_angle_-1.00.jpg")