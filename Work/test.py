from roboflow import Roboflow
rf = Roboflow(api_key="2AG2ctUU8KxikJI5kWf1")
project = rf.workspace().project("wooden-pallets-detection")
model = project.version(5).model

# # infer on a local image
print(model.predict("test.jpg", confidence=20, overlap=30).json())

# visualize your prediction
model.predict("test.jpg", confidence=10, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())