import numpy as np
import cv2
import csv
import time

# CAN UPDATE THIS IN THE FUTURE TO CHOOSE WHERE ON USERS MACHINE THE OUTPUT GOES
assetDirAdd = 'assets/' 

######################################################################
# CLEAN DATA TO GET 
#####################################################################

class CleanData:
    def __init__(self, boxes, grid_width, grid_height, reference_level=0.0, center_frequency=1.0, span=100):
        self.boxset = boxes
        self.masterwidth = grid_width
        self.masterheight = grid_height
        self.mastercenter_x = grid_width/2
        self.mastercenter_y = grid_height/2
        self.results = self.boxset

        self.reference_level = reference_level
        self.center_frequency = center_frequency
        self.span = span
        
    
    ### ISOLATE THE SIGNAL FROM THE SPECTRUM ANALYZER
    def get_results(self):
        print("\tCleaning data...")
        start = time.time()

        for box in self.results:
            hours = int(0)
            seconds = int(box[0] % 60)
            minutes = int(box[0]//60)
            if minutes != 0: hours = int(minutes//60)
            
            box[0] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"


            if box[5] < self.masterwidth*0.10: del box
            elif box[6] < self.masterheight*0.10: del box
            elif box[5] > self.masterwidth*0.90 and box[6] > self.masterheight*0.90: del box
            else:            
                if box[6] > self.masterheight* 0.80:
                    #AMPLITUDE
                    box.append(box[6]/self.masterheight)
                else:
                    box.append(0)

                if box[5] > self.masterwidth*0.70:
                    #FREQUENCY --> { [(x2 - x1 / 2) - grid_center_x] / total width } * user_input_center_frequency_value
                    box.append(((((box[3]-box[1])/2) - self.mastercenter_x) / self.masterwidth) * self.center_frequency )
                else:
                    box.append(0)
        
        end = time.time()
        print("\n\t>>> Cleaning the data took " + str(end-start) + "s\n")

        self.write_csv()

    def write_csv(self):
        # WRITE BOUNDING BOX INFORMATION TO CSV
        print("\tPrinting results to CSV...")
        start = time.time()

        header = ['timestamp', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'amplitude', 'frequency']

        with open(assetDirAdd + 'out.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for box in self.results:
                writer.writerow(box)

        end = time.time()
        print("\n\t>>> Printing results to CSV took " + str(end-start) + "s\n")