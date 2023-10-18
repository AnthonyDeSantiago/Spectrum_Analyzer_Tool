import numpy as np
import cv2
import csv
import time
import copy

# CAN UPDATE THIS IN THE FUTURE TO CHOOSE WHERE ON USERS MACHINE THE OUTPUT GOES
assetDirAdd = 'assets/' 

######################################################################
# CLEAN DATA TO GET 
#####################################################################

class CleanData:
    def __init__(self, boxes, grid_width, grid_height, reference_level=0.0, center_frequency=1.0, span=100, IOC=-10.0):
        self.boxset = copy.deepcopy(boxes)
        self.masterwidth = grid_width
        self.masterheight = grid_height
        self.mastercenter_x = grid_width/2
        self.mastercenter_y = grid_height/2
        self.results = copy.deepcopy(boxes)

        self.reference_level = reference_level
        self.center_frequency = center_frequency 
        self.span = span / 1000 # < -- convert MHz to GHz
        self.IOC = IOC
        
    
    ### ISOLATE THE SIGNAL FROM THE SPECTRUM ANALYZER
    def get_results(self):
        print("\tCleaning data...")
        start = time.time()

        for box in self.results:
            if box[5] < self.masterwidth*0.10: del box
            elif box[6] < self.masterheight*0.10: del box
            elif box[5] > self.masterwidth*0.90 and box[6] > self.masterheight*0.90: del box
            else:            
                if box[6] > self.masterheight * 0.09:
                    #MAX POWER ---- STILL NEED TO DO SOMETHING TO HANDLE IF dB GO BELOW CENTER LINE
                    max_power_percentage = (box[6]/self.masterheight) 
                    #print("max_power_percentage >>> "+ str(max_power_percentage))
                    max_power_height = 10 * self.IOC # <-- assuming grid = in blocks of 10
                    #print("max_power_height >>> "+ str(-70.0 + (max_power_percentage * max_power_height)) + "dB")
                    box.append(-70.0 - (max_power_percentage * max_power_height)) #<-- -70 is the bottom center line, conver to auto find later
                else:
                    box.append(0)

                if box[5] > self.masterwidth*0.70:
                    #FREQUENCY --> ( { [(x2 - x1) / 2) - grid_center_x] / total width } * 1/2 span ) + user_input_center_frequency_value
                    center_of_box_x = (box[3]-box[1])/2
                    x_of_box_center_relative_to_grid_center = center_of_box_x - self.mastercenter_x
                    percent_location_freq = x_of_box_center_relative_to_grid_center / self.masterwidth
                    center_of_span = (self.span  + self.center_frequency)/2
                    center_exists = percent_location_freq * center_of_span
                    freq = 0
                    if center_exists != 0: freq = self.center_frequency
                    box.append(freq) 
                else:
                    box.append(0)
        
        for box in self.results:
            hours = int(0)
            seconds = int(box[0] % 60)
            minutes = int(box[0]//60)
            if minutes != 0: hours = int(minutes//60)
            
            box[0] = f"{hours:02d}:{minutes:02d}:{seconds:02d}" #TIMESTAMP
            
            box[1] = f'({box[1]}. {box[2]})' #TOP LEFT COORDINATE
            box[2] = f'({box[3]}. {box[4]})' #BOTTOM RIGHT COORDINATE
            box[3] = box[5] #WIDTH
            box[4] = box[6] #HEIGHT

            #print(box)

            if len(box) >= 8:
                box[5] = box[7] #MAX POWER (dB)
                box[6] = box[8]
                del box[7] 
                del box[7] 
            else:
                box[5] = 0
                box[6] = 0
                

        second_counter = 0

        #print(self.boxset[0])
        #print(self.results[0])

        for i in range(int(self.boxset[-1][0])):
            hours = int(0)
            seconds = int(i % 60)
            minutes = int(i//60)
            if minutes != 0: hours = int(minutes//60)
            
            test = f"{hours:02d}:{minutes:02d}:{seconds:02d}" #TIMESTAMP

            for box in self.results:
                if box[0] == test:
                    second_counter += 1
            
            if second_counter == 0:
                self.results.append([test,0,0,0,0,0,0])
    
        print(self.results)
        self.results.sort()

        end = time.time()
        print("\n\t>>> Cleaning the data took " + str(end-start) + "s\n")

        self.write_csv()

    def write_csv(self):
        # WRITE BOUNDING BOX INFORMATION TO CSV
        print("\tPrinting results to CSV...")
        start = time.time()

        header = ['timestamp', 'top left box coord', 'bottom right box coord', 'box width', 'box height', 'max power', 'frequency']

        with open(assetDirAdd + 'out.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for box in self.results:
                writer.writerow(box)

        end = time.time()
        print("\n\t>>> Printing results to CSV took " + str(end-start) + "s\n")