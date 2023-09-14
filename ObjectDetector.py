from ultralytics import YOLO

class ObjectDetector:
    model = YOLO()
    def __init__(self, model):
        self.model = model

    def setModel(self, model):
        self.model = model

    def getDetections(self, image, imgsz):
        detections = self.model(image, imgsz=imgsz)[0]
        return detections
    
    def getBoundingBoxes(self, image, imgsz):
        detections = self.getDetections(image=image, imgsz=imgsz)
        data = detections.boxes.data.tolist()
        return data
    
    def getBoundingBoxesOfType(self, image, imgsz, classID):
        boxes = self.getBoundingBoxes(image, imgsz)
        boxesOfType = []
        for box in boxes:
            if box[5] == classID:
                boxesOfType.append(box)
        return boxesOfType
    
    def getCroppedDetections(self, image, imgsz):
        boxes = self.getBoundingBoxes(image, imgsz)
        crops = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = map(int, box)
            crops.append(image[y1:y2, x1:x2])
        return crops

    def getCroppedDetectionsOfType(self, image, imgsz, classID):
        boxes = self.getBoundingBoxesOfType(image, imgsz, classID)
        crops = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = map(int, box)
            crops.append(image[y1:y2, x1:x2])

        return crops
