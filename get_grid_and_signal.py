import numpy as np
import cv2
import csv

assetDirAdd = 'C:/Users/reave/PycharmProjects/SAT/assets/'


######################################################################
# ANALYSIS MODULE TO GET THE GRID USING CV2
#####################################################################

class GetGridCV2:

    # video_name MUST INCLUDE FULL NAME (W/FILE EXTENSION)
    def __init__(self, video_name, fps=30):
        self.video_name = str(video_name)
        self.fps = int(fps)
        self.pathAdd = assetDirAdd + video_name
        self.cap = cv2.VideoCapture(self.pathAdd)
        self.x, self.y, self.w, self.h = 0, 0, 0, 0

    ### GET A CROPPED/GAUSSIAN BLUR VIDEO OF THE SPECTRUM ANALYZER GRID
    def get_grid(self):
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        size = (frame_width, frame_height)

        writer_blur = cv2.VideoWriter(assetDirAdd + 'Sample_blur.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size,
                                      0)

        # will hold list of best contour bounding dimensions, for cropping
        xsum, ysum, wsum, hsum = [], [], [], []

        while True:
            ret, image = self.cap.read()

            if not ret:
                break

            # Convert to Gray Scale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, size)

            # Apply Gaussian Blur - write file in this format to mp4 in asset folder
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            blur = cv2.resize(blur, size)
            writer_blur.write(blur)

            # Apply Otsu's Thresholding
            ret, thresh = cv2.threshold(blur, 50, 350, cv2.THRESH_OTSU)
            thresh = cv2.resize(thresh, size)

            # Find best contour to fit the grid
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            c = 0
            for i in contours:
                area = cv2.contourArea(i)
                if area > 1000:
                    if area > max_area:
                        max_area = area
                        best_cnt = i
                        image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
                c += 1

            # Get dimensions of best contour box for this frame's grid and put in relevant list
            x, y, w, h = cv2.boundingRect(best_cnt)
            xsum.append(x)
            ysum.append(y)
            wsum.append(w)
            hsum.append(h)

        writer_blur.release()

        # Get average grid dimension values
        self.x = int(sum(xsum) / len(xsum))
        self.y = int(sum(ysum) / len(ysum))
        self.w = round(sum(wsum) / len(wsum))
        self.h = round(sum(hsum) / len(hsum))

        # Open new video stream from the grayscale Gaussian blur file generated above
        cap2 = cv2.VideoCapture(assetDirAdd + 'Sample_blur.mp4')
        writer_out = cv2.VideoWriter(assetDirAdd + 'Sample_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                     (self.w, self.h))

        # Crop video to average grid bounds
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            frame = frame[self.y:(self.y + self.h), self.x:(self.x + self.w)]
            writer_out.write(frame)

        # Clean up
        writer_out.release()
        cv2.destroyAllWindows()

    ###CALL GETSIGNALCV2
    def get_signal(self):
        signal = GetSignalCV2('Sample_out.mp4', 30, 4)
        signal.get_signal()
        signal.close()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


######################################################################
# ANALYSIS MODULE TO GET THE SIGNAL USING CV2
#####################################################################

class GetSignalCV2:
    def __init__(self, video_name, fps=30, consecutive_frames=4):
        self.video_name = str(video_name)
        self.fps = int(fps)
        self.pathAdd = assetDirAdd + video_name
        self.consecutive_frames = consecutive_frames
        self.boxes = []

    ### GET STATIC BACKGROUND IMAGE (as close to just the grid as possible)
    def get_background(self):
        cap = cv2.VideoCapture(self.pathAdd)

        # randomly select 500 frames for the calculating the median -- more frames = better average
        frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=500)

        # store the frames
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frames.append(frame)

        # calculate the median
        median_frame = np.median(frames, axis=0).astype(np.uint8)
        cap.release()

        return median_frame

    ### ISOLATE THE SIGNAL FROM THE SPECTRUM ANALYZER
    def get_signal(self):
        cap = cv2.VideoCapture(self.pathAdd)

        # get the video frame height and width
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        # define codec and create VideoWriter object ****FPS NEEDS TO BE CHANGED HERE, SHOULDN'T REMAIN 5****
        out = cv2.VideoWriter(assetDirAdd + "Bounded_Signal.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

        # Get static background, convert to grayscale
        background_raw = self.get_background()
        background = cv2.cvtColor(background_raw, cv2.COLOR_BGR2GRAY)

        frame_count = 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                frame_count += 1
                orig_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_count % self.consecutive_frames == 0 or frame_count == 1:
                    frame_diff_list = []

                # find the difference between current frame and base frame
                frame_diff = cv2.absdiff(gray, background)

                # thresholding to convert the frame to binary
                ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)

                # dilate the frame to get more white area to help with contouring
                dilate_frame = cv2.dilate(thres, None, iterations=2)

                # append the final result into the `frame_diff_list`
                frame_diff_list.append(dilate_frame)

                # if reached `consecutive_frame` number of frames
                if len(frame_diff_list) == self.consecutive_frames:
                    # add all the frames in the `frame_diff_list`
                    sum_frames = sum(frame_diff_list)

                    # find the contours around the white segmented areas
                    contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # draw the contours, not strictly necessary, just makes a nice video but uses time we don't need to
                    for i, cnt in enumerate(contours):
                        cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

                    for contour in contours:
                        # noise detection: only continue if contour area is less than 500...
                        if cv2.contourArea(contour) < 500:
                            continue
                        # get the xmin, ymin, width, and height coordinates from the contours
                        (x, y, w, h) = cv2.boundingRect(contour)
                        # add boundingRect to lists of coords, along with the frame count the rect is associated with
                        self.boxes.append([frame_count, x, y, x + w, y + h, w, h])
                        # draw the bounding boxes
                        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    out.write(orig_frame)

                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
            else:
                break

        # WRITE BOUNDING BOX INFORMATION TO CSV
        header = ['frame number', 'x1', 'y1', 'x2', 'y2', 'width', 'height']

        with open(assetDirAdd + 'out.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for box in self.boxes:
                writer.writerow(box)

        cap.release()

    def close(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sample = GetGridCV2('Sample_Video.mp4', 30)
    sample.get_grid()
    sample.get_signal()
    sample.close()
