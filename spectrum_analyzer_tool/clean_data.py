import numpy as np
import cv2
import csv
import time
import copy
from os import path
from datetime import datetime

# CAN UPDATE THIS IN THE FUTURE TO CHOOSE WHERE ON USERS MACHINE THE OUTPUT GOES
assetDirAdd = path.abspath(path.join(path.dirname(__file__),'assets/'))

######################################################################
# CLEAN DATA TO GET 
#####################################################################

class CleanData:
    def __init__(self, boxes, grid_width, grid_height, reference_level=0.0, center_frequency=1.0, span=100, IOC=-10.0):
        self.boxset = copy.deepcopy(boxes)
        self.grid_size_x = grid_width
        self.grid_size_y = grid_height
        self.mastercenter_x = grid_width/2
        self.mastercenter_y = grid_height/2
        self.results = copy.deepcopy(boxes)
        self.lb_freq = center_frequency - ((span / 1000)/2)
        self.ub_freq = center_frequency + ((span / 1000)/2)

        self.lb_power = -100
        self.ub_power = 0

        self.range_freq = self.ub_freq - self.lb_freq
        self.range_power = self.ub_power - self.lb_power

        self.reference_level = reference_level
        self.center_frequency = center_frequency 
        self.span = span / 1000 # < -- convert MHz to GHz
        self.IOC = IOC
        

        
    
    ### ISOLATE THE SIGNAL FROM THE SPECTRUM ANALYZER
    def get_results(self):
        print("\tCleaning data...")
        start = time.time()
        
        #print(self.results)
        #self.boxes.append([(frame_count + 1)/self.consecutive_frames, x, y, x + w, y + h, w, h])
        #                        0                                     1  2     3      4   5  6

#        for box in self.results:
#            if box[5] < self.grid_size_x*0.10: del box
#            elif box[6] < self.grid_size_y*0.10: del box
#            elif box[5] > self.grid_size_x*0.90 and box[6] > self.grid_size_y*0.90: del box
#            else:            
#                if box[6] > self.grid_size_y * 0.09:
#                    estimated_power = ((box[2]) / self.grid_size_y) * self.range_power + self.lb_power
#                    box.append(estimated_power) 
#                else:
#                    box.append(0)#
#
#                if box[5] > self.grid_size_x*0.70:
#                    # midpoint = x2 - (x2 - x1) // 2
#                    midpoint = box[3] - (box[3] - box[1]) // 2
#                    estimated_center_frequency = ((midpoint) / self.grid_size_x) * self.range_freq + self.lb_freq
#                    box.append(estimated_center_frequency) 
#                else:
#                    box.append(0)
#            
#                if (box[7] == 0) and (box[8] != 0): del box

        for box in self.results:
            

            #if box[5] < self.grid_size_x*0.05: 
            #    del box
            #    print("box[5] < self.grid_size_x*0.05")
            if box[6] > self.grid_size_y*0.60: 
                del box
            elif box[6] < self.grid_size_y*0.035:
                del box
            elif box[2] < self.grid_size_y*0.03: 
                del box
            elif box[1] < self.grid_size_x*0.1: 
                del box
            #elif box[6] < self.grid_size_y*0.10: del box
            #elif box[5] > self.grid_size_x*0.90 and box[6] > self.grid_size_y*0.90: del box
            else:            
                if (box[4] > (self.grid_size_y * 0.2)) and (box[3] > (self.grid_size_x*0.40)):
                    #estimated_power =  (y1 / width of grid) * range_power + lb_power
                    estimated_power = ((box[2]) / self.grid_size_y) * self.range_power + self.lb_power
                    box.append(estimated_power) #<-- -70 is the bottom center line, conver to auto find later
                #else:
                #    box.append(0)

                #if (box[3] - (self.grid_size_x*0.50)) > 0:
                    # midpoint = x2 - (x2 - x1) // 2
                    midpoint = box[3] - (box[3] - box[1]) // 2
                    estimated_center_frequency = ((midpoint) / self.grid_size_x) * self.range_freq + self.lb_freq
                    box.append(estimated_center_frequency) 
                else:
                    box.append(0)
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
    
        #print(self.results)
        self.results.sort()

        end = time.time()
        print("\n\t>>> Cleaning the data took " + str(end-start) + "s\n")

        self.write_csv()

    def write_csv(self):
        # WRITE BOUNDING BOX INFORMATION TO CSV
        print("\tPrinting results to CSV...")
        start = time.time()

        timestamp = "00"
        tempboxes = []

        for box in self.results:
            if timestamp == box[0]:
                if box[5] > 0 or box[6] > 0:
                    tempboxes.append(box)
            else:
                tempboxes.append(box)
            timestamp = box[0]



        header = ['timestamp', 'frequency', 'max power']

        name_time = datetime.now()
        output_filename = "Unsupervised_Out_"+ str(name_time.month) +"_"+ str(name_time.day)+"_"+str(name_time.year) +"_"+ str(name_time.hour)+"_"+ str(name_time.minute)+ "_" + str(name_time.second) + ".csv"

        with open(path.abspath(path.join('.',output_filename)), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for box in tempboxes:
                tempbox = []
                tempbox.append(box[0])
                tempbox.append(box[6])
                tempbox.append(box[5])
                writer.writerow(tempbox)

        end = time.time()
        print("\n\t>>> Printing results to CSV took " + str(end-start) + "s\n")