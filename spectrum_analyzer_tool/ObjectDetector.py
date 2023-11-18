from ultralytics import YOLO
import cv2

class ObjectDetector:
    """
    A class for performing object detection using YOLOv8

    Basically, this is to segregate all or atleast most of the functionality of detecting objects in an image.
    This way we can instantiate several ObjectDectector objects that each using their own model for detecting certain things.
    For example, one could detect the spectrum analyzer screen every ith frame another could grab some other element
    every jth frame etc.

    Attributes:
        model: The YOLO model for object detection.

    Methods:
        __init__(model): Initialize the ObjectDetector with a YOLO model.
        setModel(model): Set a new YOLO model for the detector.
        getDetections(image, imgsz): Get object detections in the given image.
        getBoundingBoxes(image, imgsz): Get bounding boxes of detected objects.
        getBoundingBoxesOfType(image, imgsz, classID): Get bounding boxes of objects of a specific class.
        getCroppedDetections(image, imgsz): Get cropped detections from the image.
        getCroppedDetectionsOfType(image, imgsz, classID): Get cropped detections of objects of a specific class.
    """
    model = YOLO()
    def __init__(self, model, imgz):
        """
        Initialize the ObjectDetector with a YOLO model

        Args:
            model(Any): The trained model to be used for object detection
        """
        self.model = model
        self.imgsz = imgz

    def setModel(self, model):
        """
        A setter method for the trained model of this object

        Args:
            model (Any): This can be .pt, .onnx, openvino, tflite, etc
        """
        self.model = model

    def getDetections(self, image):
        """
        Get object detections in the given image.

        Args:
            image: The input image on which object detection will be performed.
            imgsz: Here put the size of the img the model was trained on.

        Returns:
            detections: Detected objects in the image.
        """
        detections = self.model(image, imgsz=self.imgsz)[0]
        return detections
    
    def getBoundingBoxes(self, image):
        """
        Get bounding boxes of detected objects in the image.

        Args:
            image: The input image containing objects.
            imgsz: Here put the size of the img the model was trained on.

        Returns:
            data: List of bounding box coordinates.
        """
        detections = self.getDetections(image=image)
        data = detections.boxes.data.tolist()
        return data
    
    def getBoundingBoxesOfType(self, image, classID):
        """
        Get bounding boxes of objects of a specific class in the image.

        Args:
            image: The input image containing objects.
            imgsz: Here put the size of the img the model was trained on.
            classID: The ID of the class to filter for.

        Returns:
            boxesOfType: List of bounding box coordinates for the specified class.
        """
        boxes = self.getBoundingBoxes(image)
        boxesOfType = []
        for box in boxes:
            if box[5] == classID:
                boxesOfType.append(box)
        return boxesOfType
    
    def getCroppedDetections(self, image):
        """
        Get cropped detections from the image.

        Args:
            image: The input image containing objects.
            imgsz: Here put the size of the img the model was trained on.

        Returns:
            crops: List of cropped regions of detected objects.
        """
        boxes = self.getBoundingBoxes(image)
        crops = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = map(int, box)
            crops.append(image[y1:y2, x1:x2])
        return crops

    def getCroppedDetectionsOfType(self, image, classID):
        """
        Get cropped detections of objects of a specific class from the image.

        Args:
            image: The input image containing objects.
            imgsz: Here put the size of the img the model was trained on.
            classID: The ID of the class to filter for.

        Returns:
            crops: List of cropped regions of objects of the specified class.
        """
        boxes = self.getBoundingBoxesOfType(image, classID)
        crops = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = map(int, box)
            crops.append(image[y1:y2, x1:x2])

        return crops
    
    def drawBoxes(self, image, threshold):
        boxes = self.getBoundingBoxes(image)
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = map(int, box)
            if box[4] > threshold:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image
