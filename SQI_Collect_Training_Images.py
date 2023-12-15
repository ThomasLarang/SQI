from serial import Serial, SEVENBITS, EIGHTBITS, STOPBITS_ONE, PARITY_NONE, rs485
from pypylon import pylon
import time, csv, random, cv2
from datetime import datetime

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

winTitle = "SQI"
image_grab_delay_time = 0.3
batch_ID = 39
test_size = 1
print('Batch: ' + str(batch_ID))
print('Test size: ' + str(test_size))
cameraExposure = 80000
margin_X = 200
margin_Y = 300
try:
    StepperComPort = Serial('/dev/ttyACM1', baudrate=115200)
except:
    StepperComPort = Serial('/dev/ttyACM0', baudrate=115200)
with open('slider_positions_Chamber_Reduced.csv', newline='')as f:
    reader = csv.reader(f, delimiter=';')
    slider_positions_X = next(reader)
    slider_positions_Y = next(reader)
    print(slider_positions_X[0])

def LIGHT(x):
    print("Sending message to light")
    StepperComPort.write(bytes(x, 'utf-8'))
    print("Message to light sent")
    

def Stepper_write(x):
    StepperComPort.write(bytes(x, 'utf-8'))

    print("Sent message: " +str(x))
    print("length of message sent: " + str(len(x)))
    time.sleep(0.05)
    r = StepperComPort.read(len(x)+2).decode('utf-8')
    time.sleep(0.05);
    print("Returned message: " + str(r))
#    if r == x:
#        print("Sent message: ok")
 
Stepper_write('Home')
LIGHT('AMBIENT_LIGHT_OFF')
time.sleep(0.2)
LIGHT('LASER_LIGHT_OFF')
time.sleep(0.2)
LIGHT('BLUE_LIGHT_OFF')
time.sleep(0.2)
LIGHT('AMBIENT_LIGHT_ON')
time.sleep(0.1)


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(cameraExposure)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

base_options = core.BaseOptions(file_name='detect_with_metadata.tflite', use_coral=False, num_threads=4)
detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

for n in range(0, int(test_size)):

    Stepper_write('Y_M'+str(slider_positions_Y[n]))
    time.sleep(image_grab_delay_time)
    for l in range(0,13):
        
        chamberstart = datetime.now()
        start = datetime.now() 
        if(n % 2) == 0: #n is even
            i = l
        else:
            i = 12 - l
        Stepper_write('X_M'+str(slider_positions_X[i]))
        
        time.sleep(image_grab_delay_time)#To let vibration settle and clear out irrelevant images buffered in camera
        stop = datetime.now() 
        print("Chamber move time: " +str(stop-start))

        start = datetime.now()
        grabResult = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
         
        ambient=converter.Convert(grabResult)
        ambient = ambient.GetArray()
        rgb_image = cv2.cvtColor((ambient), cv2.COLOR_BGR2RGB)
        stop = datetime.now()
        print("Fetch image time: " +str(stop-start))
        start = datetime.now()
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(input_tensor)
        y0 = detection_result.detections[0].bounding_box.origin_y
        y1 = detection_result.detections[0].bounding_box.origin_y+detection_result.detections[0].bounding_box.height
        x0 = detection_result.detections[0].bounding_box.origin_x
        x1 = detection_result.detections[0].bounding_box.origin_x+detection_result.detections[0].bounding_box.width
        if bool(detection_result.detections) == True:
            ambient_cropped =  ambient[y0-margin_Y:margin_Y+y1,x0-margin_X:margin_X+x1]
            anomaly_check_img = cv2.cvtColor(cv2.resize(ambient_cropped, (144,144)), cv2.COLOR_BGR2GRAY)
            anomaly_check_imgUL = anomaly_check_img[0:32, 14:46]
            anomaly_check_imgUR = anomaly_check_img[0:32, 98:130]
            anomaly_check_imgLL = anomaly_check_img[112:144, 14:46]
            anomaly_check_imgLR = anomaly_check_img[112:144, 98:130]
            cv2.imwrite('/home/sqi/SQI_unsorted_training_images/UL/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgUL) #Activate when collecting training images
            cv2.imwrite('/home/sqi/SQI_unsorted_training_images/UR/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgUR) #Activate when collecting training images
            cv2.imwrite('/home/sqi/SQI_unsorted_training_images/LL/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgLL) #Activate when collecting training images
            cv2.imwrite('/home/sqi/SQI_unsorted_training_images/LR/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgLR) #Activate when collecting training images
            cv2.imwrite('/home/sqi/SQI_unsorted_training_images/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_img) #Activate when collecting training images
        
        cv2.imshow(winTitle, cv2.resize(ambient, (1006, 759)))
        cv2.waitKey(5)
        
Stepper_write('Home')
print("homed")
camera.StopGrabbing()
camera.Close()
Stepper_write('Disable')
time.sleep(0.2)
LIGHT('BLUE_LIGHT_OFF')
time.sleep(0.2)
LIGHT('LASER_LIGHT_OFF')
time.sleep(0.2)
LIGHT('AMBIENT_LIGHT_OFF')   