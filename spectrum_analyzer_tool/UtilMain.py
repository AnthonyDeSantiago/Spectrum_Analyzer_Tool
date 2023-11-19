from .Util import*
from os import path

imgPath = path.abspath(path.join(path.dirname(__file__),'assets/graph_sample.png'))

util.getOCR(imgPath)
util.getReferenceLevel(imgPath)
util.getCenter(imgPath)
util.getSpan(imgPath)



