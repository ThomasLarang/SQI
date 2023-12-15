from PIL import Image
from pypylon import pylon
import os, os.path, cv2, time, csv
from datetime import datetime
import numpy as np
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
from RpiMotorLib import RpiMotorLib

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
#from tensorflow.keras.models import Sequential
"""
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
"""
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

Startup_start = datetime.now() 

Ambient = 4
Laser = 2
Blue = 3
Home_X = 23
Home_Y = 18
Home_Z = 24
X_step = 5
X_direction = 6
X_en = 13
Y_step = 16
Y_direction = 20
Y_en = 21
Z_direction = 17
Z_step = 27
Z_en = 22
X_GPIO_pins = (14,15,18)
Y_GPIO_pins = (14,15,18)
Z_GPIO_pins = (14,15,18)

GPIO.setup(Ambient, GPIO.OUT)
GPIO.setup(Laser, GPIO.OUT)
GPIO.setup(Blue, GPIO.OUT)
GPIO.setup(Home_X, GPIO.IN)
GPIO.setup(Home_Y, GPIO.IN)
GPIO.setup(Home_Z, GPIO.IN)
GPIO.setup(X_en, GPIO.OUT)
GPIO.setup(Y_en, GPIO.OUT)
GPIO.setup(Z_en, GPIO.OUT)
GPIO.output(Ambient, GPIO.LOW)
X_motor = RpiMotorLib.A4988Nema(X_direction, X_step, Z_GPIO_pins, "A4988")
Y_motor = RpiMotorLib.A4988Nema(Y_direction, Y_step, Z_GPIO_pins, "A4988")
Z_motor = RpiMotorLib.A4988Nema(Z_direction, Z_step, Z_GPIO_pins, "A4988")
 
reconstruction_error_threshold_ch = 0.007
reconstruction_error_threshold_ul = 0.010 #Default 0.005
reconstruction_error_threshold_ur = 0.0050 #Default 0.005

winTitle = "SQI"
SIZE = 144 # px size of each side in anomaly check image
checkpoint_path_ch = '/home/sqi/SQI/chamber_recognition_model_lite/Anomaly_Checkpoints/SQI/cp.ckpt'
checkpoint_path_ul = '/home/sqi/SQI/chamber_recognition_model_lite/Anomaly_Checkpoints/SQI/UL_cp/UL_cp.ckpt'
checkpoint_path_ur = '/home/sqi/SQI/chamber_recognition_model_lite/Anomaly_Checkpoints/SQI/UR_cp/UR_cp.ckpt'

checkpoint_dir_ch = os.path.dirname(checkpoint_path_ch)
checkpoint_dir_ul = os.path.dirname(checkpoint_path_ul)
checkpoint_dir_ur = os.path.dirname(checkpoint_path_ur)

latent_space_ch= np.load('/home/sqi/SQI/chamber_recognition_model_lite/Anomaly_Checkpoints/SQI/latent_space.npy')
latent_space_ul= np.load('/home/sqi/SQI/chamber_recognition_model_lite/Anomaly_Checkpoints/SQI/UL_cp/latent_space.npy')
latent_space_ur= np.load('/home/sqi/SQI/chamber_recognition_model_lite/Anomaly_Checkpoints/SQI/UR_cp/latent_space.npy')
out_vector_shape_ch = int(9*9*8)
out_vector_shape_ul = int(4*4*8)
out_vector_shape_ur = int(4*4*8)
encoded_images_vector_ch = [np.reshape(img, (out_vector_shape_ch)) for img in latent_space_ch]
encoded_images_vector_ul = [np.reshape(img, (out_vector_shape_ul)) for img in latent_space_ul]
encoded_images_vector_ur = [np.reshape(img, (out_vector_shape_ur)) for img in latent_space_ur]

FAIL = 0
PASS = 0
SIZE = 144
margin_X = 200
margin_Y = 300
num_classes = 4
img_height = 180
img_width = 180

height = 3036
width = 4024
ar = width/height

peak_interval = 2500 #Distance in px between chamber walls

BGR_OK = (0,240,0)
BGR_NOK = (0,0,240)
RGB_OK = (0,255, 0)
RGB_NOK = (255,0,0)

cameraExposure = 80000
h=670
w=506
w_Im = int((12*h))
h_Im =  2*w+50
image_grab_delay_time = 0.3


with open('slider_positions_Chamber_Reduced.csv', newline='')as f:
    reader = csv.reader(f, delimiter=';')
    slider_positions_X = next(reader)
    slider_positions_Y = next(reader)
    slider_positions_Z = next(reader)

X_abs_pos = 0
Y_abs_pos = 0
Z_abs_pos = 0

def homing():
    print("Homing")
    if GPIO.input(Home_X) == True:
        print("X closed")
        X_motor.motor_go(False, "Full" , 50, .0003, False, .0003)
    if GPIO.input(Home_Y) == True:
        print("Y closed")
        Y_motor.motor_go(False, "Full" , 50, .0003, False, .0003)
    if GPIO.input(Home_Z) == True:
        print("Z closed")
        Z_motor.motor_go(True, "Full" , 50, .0005, False, .0005)
        
    if GPIO.input(Home_X) == False:
        print("Homing X")
        while GPIO.input(Home_X) == False:
            X_motor.motor_go(True, "Full" , 5, .0002, False, .0004)
        X_motor.motor_go(False, "Full" , 100, .0002, False, .0004)
        while GPIO.input(Home_X) == False:
            X_motor.motor_go(True, "Full" , 1, .0008, False, .0004)
    if GPIO.input(Home_Y) == False:
        print("Homing Y")
        while GPIO.input(Home_Y) == False:
            Y_motor.motor_go(True, "Full" , 5, .0002, False, .0004)
        Y_motor.motor_go(False, "Full" , 100, .0002, False, .0004)
        while GPIO.input(Home_Y) == False:
            Y_motor.motor_go(True, "Full" , 1, .0008, False, .0004)
    print("Homing Z")
    while GPIO.input(Home_Z) == False:    
        Z_motor.motor_go(False, "Full" , 1, .0004, False, .0005)
    X_abs_pos = 0
    Y_abs_pos = 0
    Z_abs_pos = 0
    return Y_abs_pos, X_abs_pos, Z_abs_pos   
    time.sleep(0.1)
    print("X, Y and Z axis are homed")

def LIGHT(light):
    if light == "AMBIENT_LIGHT_ON":
        GPIO.output(Ambient, GPIO.HIGH)
    elif light == "AMBIENT_LIGHT_OFF":
        GPIO.output(Ambient, GPIO.LOW)
    elif light == "LASER_LIGHT_ON":
        GPIO.output(Laser, GPIO.HIGH)
    elif light == "Laser_LIGHT_OFF":
        GPIO.output(Laser, GPIO.LOW)
    if light == "BLUE_LIGHT_ON":
        GPIO.output(Blue, GPIO.HIGH)
    elif light == "BLUE_LIGHT_OFF":
        GPIO.output(Blue, GPIO.LOW) 

def check_anomaly_ch(img):
    img = img / 255.
    img = img[:,:, np.newaxis]
    img = img[np.newaxis, :]
    start_CA = datetime.now()
    
    reconstruction = model_ch.predict([[img]])
    reconstruction_error = model_ch.evaluate([reconstruction],[[img]], batch_size = 1)[0]
    stop_CA = datetime.now()
    print("Anomaly check time: " +str(stop_CA-start_CA))
    if reconstruction_error > reconstruction_error_threshold_ch:
        Anomaly_message = "Chamber Anomaly"
    else:
        Anomaly_message = "Chamber OK"
    return Anomaly_message, reconstruction_error

def check_anomaly_ul(img):
    print("ul")
    img = img / 255.
    img = img[:,:, np.newaxis]
    img = img[np.newaxis, :]
    reconstruction = model_ul.predict([[img]])
    reconstruction_error = model_ul.evaluate([reconstruction],[[img]], batch_size = 1)[0]
    if  reconstruction_error > reconstruction_error_threshold_ul:
        Anomaly_message = "UL Anomaly"
    else:
        Anomaly_message = "UL OK"
    return Anomaly_message, reconstruction_error

def check_anomaly_ur(img):
    print("ur")
    img = img / 255.
    img = img[:,:, np.newaxis]
    img = img[np.newaxis, :]
    reconstruction = model_ur.predict([[img]])
    reconstruction_error = model_ur.evaluate([reconstruction],[[img]], batch_size = 1)[0]
    #print("Density: " +str(density) + "Recons.Error: " + str(reconstruction_error))
    if reconstruction_error > reconstruction_error_threshold_ur: #density < density_threshold or
        Anomaly_message = "UR Anomaly"
    else:
        Anomaly_message = "UR OK"
    return Anomaly_message, reconstruction_error

GPIO.output(X_en, GPIO.LOW)
GPIO.output(Y_en, GPIO.HIGH)
GPIO.output(Z_en, GPIO.LOW)

homing()
time.sleep(0.5)

#Setup parameters start
batch_ID = 0
sequence_ID = str("F")
test_size = 3
print('Batch: ' + str(batch_ID))
print('Sequence: ' + str(sequence_ID))
print('Test size: ' + str(test_size))

savedir = r'/home/sqi/SQI_results' #%(batch_ID)
if not os.path.exists(savedir):
    os.mkdir(savedir)


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

def create_model_ch():
    #Encoder
    model_ch = tf.keras.Sequential()
    model_ch.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 1))) #, weights=encoder_weights.layers[0].get_weights() 
    model_ch.add(MaxPooling2D((2, 2), padding='same'))
    model_ch.add(Conv2D(32, (3, 3), activation='relu', padding='same')) #, weights=model.layers[2].get_weights()
    model_ch.add(MaxPooling2D((2, 2), padding='same'))
    model_ch.add(Conv2D(16, (3, 3), activation='relu', padding='same')) #, weights=model.layers[2].get_weights()
    model_ch.add(MaxPooling2D((2, 2), padding='same'))
    model_ch.add(Conv2D(8, (3, 3), activation='relu', padding='same')) #, weights=model.layers[4].get_weights()
    model_ch.add(MaxPooling2D((2, 2), padding='same'))
    #Decoder
    model_ch.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    model_ch.add(UpSampling2D((2,2)))
    model_ch.add(Conv2D(16,(3,3), activation='relu', padding='same'))
    model_ch.add(UpSampling2D((2,2)))
    model_ch.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model_ch.add(UpSampling2D((2,2)))
    model_ch.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    model_ch.add(UpSampling2D((2,2)))
    model_ch.add(Conv2D(1,(3,3), activation='sigmoid', padding='same'))
    model_ch.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])   
    return model_ch

model_ch = create_model_ch()
latest_ch = tf.train.latest_checkpoint(checkpoint_dir_ch)
model_ch.load_weights(latest_ch)


def create_model_ul():
    #Encoder
    model_ul = tf.keras.Sequential()
    model_ul.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1))) 
    model_ul.add(MaxPooling2D((2, 2), padding='same'))
    model_ul.add(Conv2D(16, (3, 3), activation='relu', padding='same')) 
    model_ul.add(MaxPooling2D((2, 2), padding='same'))
    model_ul.add(Conv2D(8, (3, 3), activation='relu', padding='same')) 
    model_ul.add(MaxPooling2D((2, 2), padding='same'))
    #Decoder
    model_ul.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    model_ul.add(UpSampling2D((2,2)))
    model_ul.add(Conv2D(16,(3,3), activation='relu', padding='same'))
    model_ul.add(UpSampling2D((2,2)))
    model_ul.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model_ul.add(UpSampling2D((2,2)))
    model_ul.add(Conv2D(1,(3,3), activation='sigmoid', padding='same'))
    model_ul.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])   
    return model_ul

model_ul = create_model_ul()
latest_ul = tf.train.latest_checkpoint(checkpoint_dir_ul)
model_ul.load_weights(latest_ul)

def create_model_ur():
    #Encoder
    model_ur = tf.keras.Sequential()
    model_ur.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1))) 
    model_ur.add(MaxPooling2D((2, 2), padding='same'))
    model_ur.add(Conv2D(16, (3, 3), activation='relu', padding='same')) 
    model_ur.add(MaxPooling2D((2, 2), padding='same'))
    model_ur.add(Conv2D(8, (3, 3), activation='relu', padding='same')) 
    model_ur.add(MaxPooling2D((2, 2), padding='same'))    

    #Decoder
    model_ur.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    model_ur.add(UpSampling2D((2,2)))
    model_ur.add(Conv2D(16,(3,3), activation='relu', padding='same'))
    model_ur.add(UpSampling2D((2,2)))
    model_ur.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model_ur.add(UpSampling2D((2,2)))
    model_ur.add(Conv2D(1,(3,3), activation='sigmoid', padding='same'))
    model_ur.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])   
    return model_ur

model_ur = create_model_ur()
latest_ur = tf.train.latest_checkpoint(checkpoint_dir_ur)
model_ur.load_weights(latest_ur)


LIGHT('AMBIENT_LIGHT_OFF')
time.sleep(0.2)
LIGHT('LASER_LIGHT_OFF')
time.sleep(0.2)
LIGHT('BLUE_LIGHT_OFF')
time.sleep(0.2)
LIGHT('AMBIENT_LIGHT_ON')
time.sleep(0.1)

window_scale = 0.75
t_rack_margin_x = 120
t_rack_margin_y = 120
t_rack_spacing_x = 500
sl_w = 384
ch_w = 8
ch_h = 13
ch_w_spacing = int((sl_w-17)/13)
sl_h = 13
sl_y_spacing = 67
sl_x_offset = 134
sl_y_offset =0
ch_x_offset = 20
ch_RGB_grey = (140, 140, 140)
sl_RGB_grey = (120, 120, 120)
display_height = 880
display_width = 1850
RGB_filled_ch = (194, 194, 194)
RGB_Transport_Rack = (165, 165, 165)
RGB_Transport_Rack_Fill = (220, 220, 220)

display = np.zeros((display_height, display_width, 3), np.uint8)
display[0:display_height,0:display_width] = (200, 200, 200) # Red in BGR format

t_rack_outer_pts =[(0, 14), (1, 10), (2, 6), (5, 4), (8, 1), (14, 0), (413, 0), (418, 1), (422, 4), (424, 6), (426, 10), (426, 14), (426, 497), (426, 502), (424, 506), (422, 509), (418, 510), (413, 512), (14, 512), (10, 511), (6, 509), (4, 507), (1, 504), (0, 498)]
t_rack_inner_pts = [(37, 40), (38, 39), (388, 39), (390, 40), (390, 65), (405, 65), (405, 78), (390, 78), (390, 40), (390, 129), (405, 129), (405, 142), (390, 142), (390, 165), (405, 165), (405, 178), (390, 178), (390, 229), (405, 229), (405, 242), (390, 242), (390, 270), (405, 270), (405, 283), (390, 283), (390, 334), (405, 334), (405, 347), (390, 347), (390, 370), (405, 370), (405, 383), (390, 383), (390, 434), (405, 434), (405, 447), (390, 447), (390, 471), (388, 473), (38, 473), (37, 471), (37, 447), (21, 447), (21, 434), (37, 434), (37, 383), (21, 383), (21, 370), (37, 370), (37, 347), (21, 347), (21, 334), (37, 334), (37, 283), (21, 283), (21, 270), (37, 270), (37, 242), (21,242), (21, 229), (37, 229), (37, 178), (21, 178), (21, 165), (37, 165), (37, 142), (21, 142), (21, 129), (37, 129), (37, 78), (21, 78), (21, 65), (37, 65)]

def chamber_display(x,y, RGB, T):
    cv2.rectangle(display, (x,y), (x+ch_w, y+ch_h), RGB_filled_ch, -1)
    cv2.rectangle(display, (x,y), (x+ch_w, y+ch_h), RGB, T)
    
def slider_display(x,y, RGB):
    cv2.rectangle(display, (x,y), (x+sl_w, y+sl_h), RGB, 1)
    
def transport_rack_display(x,y,RGB):
    t_rack_inner_translate = np.array([t_rack_inner_pts])
    t_rack_outer_translate = np.array([t_rack_outer_pts])
    for i in range(len(np.array([t_rack_inner_pts][0]))):      
        t_rack_inner_translate[0][i][0] = t_rack_inner_translate[0][i][0] + x
        t_rack_inner_translate[0][i][1] = t_rack_inner_translate[0][i][1] + y
    for i in range(len(np.array([t_rack_outer_pts][0]))):      
        t_rack_outer_translate[0][i][0] = t_rack_outer_translate[0][i][0] + x
        t_rack_outer_translate[0][i][1] = t_rack_outer_translate[0][i][1] + y
    cv2.polylines(display, np.array([t_rack_inner_translate]), True, RGB, 1)
    cv2.polylines(display, np.array([t_rack_outer_translate]), True, RGB, 1)
    
for t in range(0, 3):
   transport_rack_display(int(t_rack_margin_x+t_rack_spacing_x*t),t_rack_margin_y,RGB_Transport_Rack)
slider_display(t_rack_inner_pts[43][0]+t_rack_margin_x, t_rack_inner_pts[43][1]+t_rack_margin_y, sl_RGB_grey)
slider_display(t_rack_inner_pts[47][0]+t_rack_margin_x, t_rack_inner_pts[47][1]+t_rack_margin_y, sl_RGB_grey)
slider_display(t_rack_inner_pts[51][0]+t_rack_margin_x, t_rack_inner_pts[51][1]+t_rack_margin_y, sl_RGB_grey)
slider_display(t_rack_inner_pts[55][0]+t_rack_margin_x, t_rack_inner_pts[55][1]+t_rack_margin_y, sl_RGB_grey)  
 
for l in range(0,13):   
    chamber_display(t_rack_inner_pts[43][0]+t_rack_margin_x+ch_x_offset+ ch_w_spacing*l, t_rack_inner_pts[43][1]+t_rack_margin_y, ch_RGB_grey, 1)
for l in range(0,13):   
    chamber_display(t_rack_inner_pts[47][0]+t_rack_margin_x+ch_x_offset+ ch_w_spacing*l, t_rack_inner_pts[47][1]+t_rack_margin_y, ch_RGB_grey, 1)
for l in range(0,13):   
    chamber_display(t_rack_inner_pts[51][0]+t_rack_margin_x+ch_x_offset+ ch_w_spacing*l, t_rack_inner_pts[51][1]+t_rack_margin_y, ch_RGB_grey, 1)
for l in range(0,13):   
    chamber_display(t_rack_inner_pts[55][0]+t_rack_margin_x+ch_x_offset+ ch_w_spacing*l, t_rack_inner_pts[55][1]+t_rack_margin_y, ch_RGB_grey, 1)

Z_move_rel = int(slider_positions_Z[0])- Z_abs_pos
Z_motor.motor_go(True, "Full" , Z_move_rel, .0003, False, .0003)
Z_abs_pos = int(slider_positions_Z[0])
        
cv2.imshow(winTitle, display)
cv2.waitKey(300)
s=0

Startup_stop = datetime.now()
Run_start = datetime.now()
for n in range(0, int(test_size)):
    
    sliderstart = datetime.now()
    SliderOverview = Image.new("RGB", (int(2*w+w_Im/2), int(2*h_Im)), "black")
    Y_move_rel = int(slider_positions_Y[n])- Y_abs_pos
    Y_motor.motor_go(False, "Full" , Y_move_rel, .0002, False, .0003)
    Y_abs_pos = int(slider_positions_Y[n])
    
    time.sleep(image_grab_delay_time)
       
    for l in range(0,13):
        
        chamberstart = datetime.now()
        start = datetime.now() 
        if(n % 2) == 0: #n is even
            i = l
            X_direction = False
            X_move_rel = int(slider_positions_X[i])- X_abs_pos
        else:
            X_direction = True
            i = 12 - l
            X_move_rel = X_abs_pos -int(slider_positions_X[i])
        print(X_move_rel)
        X_motor.motor_go(X_direction, "Full" , X_move_rel, .0004, False, .001)
        X_abs_pos = int(slider_positions_X[i])
        print(X_abs_pos)
        
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
        stop = datetime.now()
        print("Chamber recognition time: " + str(stop-start))
        
        if bool(detection_result.detections) == True:
            start= datetime.now()
            ambient_cropped =  ambient[y0-margin_Y:margin_Y+y1,x0-margin_X:margin_X+x1]
            anomaly_check_img = cv2.cvtColor(cv2.resize(ambient_cropped, (144,144)), cv2.COLOR_BGR2GRAY)

            anomaly_check_imgUL = anomaly_check_img[0:32, 14:46]
            anomaly_check_imgUR = anomaly_check_img[0:32, 98:130]
            anomaly_check_imgLL = anomaly_check_img[112:144, 14:46]
            anomaly_check_imgLR = anomaly_check_img[112:144, 98:130]
            #cv2.imwrite('/home/sqi/SQI_unsorted_training_images/UL/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgUL) #Activate when collecting training images
            #cv2.imwrite('/home/sqi/SQI_unsorted_training_images/UR/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgUR) #Activate when collecting training images
            #cv2.imwrite('/home/sqi/SQI_unsorted_training_images/LL/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgLL) #Activate when collecting training images
            #cv2.imwrite('/home/sqi/SQI_unsorted_training_images/LR/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_imgLR) #Activate when collecting training images
            #cv2.imwrite('/home/sqi/SQI_unsorted_training_images/B%s_Sl%s_ch%s.jpg' %(batch_ID, n, l), anomaly_check_img) #Activate when collecting training images

            #cv2.putText(ambient, "Chamber", (detection_result.detections[0].bounding_box.origin_x,detection_result.detections[0].bounding_box.origin_y),cv2.FONT_HERSHEY_SIMPLEX, 5, (200,200,200), 5)   
            cv2.putText(ambient,"X: " + str(detection_result.detections[0].bounding_box.origin_x+detection_result.detections[0].bounding_box.width/2), (50,150),cv2.FONT_HERSHEY_SIMPLEX, 5, (200,200,200), 5)
            cv2.putText(ambient,"Y: " + str(detection_result.detections[0].bounding_box.origin_y+detection_result.detections[0].bounding_box.height/2), (50,350),cv2.FONT_HERSHEY_SIMPLEX, 5, (200,200,200), 5)
            cv2.putText(ambient,"Sl : " + str(n), (3350,150),cv2.FONT_HERSHEY_SIMPLEX, 5, (200,200,200), 5)
            cv2.putText(ambient,"Ch : " + str(i), (3350,350),cv2.FONT_HERSHEY_SIMPLEX, 5, (200,200,200), 5)
            """
            cv2.imshow(winTitle, cv2.resize(rgb_image, (144, 144)))
            
            cv2.waitKey(5)
            """
            print(anomaly_check_img.shape)
            print(anomaly_check_imgUL.shape)
            print(anomaly_check_imgUR.shape)
            stop= datetime.now()
            print("Image preprocessing time: " + str(stop-start))
            start= datetime.now()
            results_ch = check_anomaly_ch(anomaly_check_img)
            results_ul = check_anomaly_ul(anomaly_check_imgUL)
            results_ur = check_anomaly_ur(anomaly_check_imgUR)
            if results_ch[0] == "Chamber OK":
                cv2.putText(ambient, str(results_ch[0]), (1000, 2750),cv2.FONT_HERSHEY_SIMPLEX, 10, RGB_OK, 15)
                cv2.putText(ambient, str(results_ch[1]), (200, 500),cv2.FONT_HERSHEY_SIMPLEX, 5, RGB_OK, 10)
            else:
                cv2.putText(ambient, str(results_ch[0]), (1000, 2750),cv2.FONT_HERSHEY_SIMPLEX, 10, RGB_NOK, 15)
                cv2.putText(ambient, str(results_ch[1]), (200, 500),cv2.FONT_HERSHEY_SIMPLEX, 5, RGB_NOK, 10)

            if results_ul[0] == "UL OK":
                cv2.rectangle(ambient, (x0-margin_X,y0-margin_Y), (x0+margin_X,y0+margin_Y), RGB_OK, 25)
                cv2.putText(ambient, '{:3f}'.format(results_ul[1]), (200, 1200),cv2.FONT_HERSHEY_SIMPLEX, 4, RGB_OK, 8)

            else:
                cv2.rectangle(ambient, (x0-margin_X,y0-margin_Y), (x0+margin_X,y0+margin_Y), RGB_NOK, 45)
                cv2.putText(ambient, '{:.3f}'.format(results_ul[1]), (200, 1200),cv2.FONT_HERSHEY_SIMPLEX, 4, RGB_NOK, 8)

            if results_ur[0] == "UR OK":
                cv2.rectangle(ambient, (x1-margin_X,y0-margin_Y), (x1+margin_X,y0+margin_Y), RGB_OK, 25)
                cv2.putText(ambient, '{:.3f}'.format(results_ur[1]), (3200, 1200),cv2.FONT_HERSHEY_SIMPLEX, 4, RGB_OK, 8)

            else:
                cv2.rectangle(ambient, (x1-margin_X,y0-margin_Y), (x1+margin_X,y0+margin_Y), RGB_NOK, 45)
                cv2.putText(ambient, '{:.3f}'.format(results_ur[1]), (3200, 1200),cv2.FONT_HERSHEY_SIMPLEX, 4, RGB_NOK, 8)
            ambient = cv2.resize(ambient, (h,w))
            
            if results_ch[0] == "Chamber OK" and results_ul[0] == "UL OK" and results_ur[0] == "UR OK":
                chamber_display(t_rack_inner_pts[43+(4*n)][0]+t_rack_margin_x+ch_x_offset+ ch_w_spacing*i, t_rack_inner_pts[43+(4*n)][1]+t_rack_margin_y, BGR_OK, 2)
                
            else:
                chamber_display(t_rack_inner_pts[43+(4*n)][0]+t_rack_margin_x+ch_x_offset+ ch_w_spacing*i, t_rack_inner_pts[43+(4*n)][1]+t_rack_margin_y, BGR_NOK, 2)             
        
            cv2.imshow(winTitle, display)
            cv2.waitKey(5)   
            
        stop = datetime.now()
        print("Object analysis time: " +str(stop-start))        

        start = datetime.now() 

        cv2.rectangle(ambient, (0,0), (h,w), (200,200,200), 3)
        if(n % 2) == 0: #n is even
            i = l
            if i < 6:
                SliderOverview.paste(Image.fromarray(ambient), (i*h, w))
            else:
                SliderOverview.paste(Image.fromarray(ambient), ((i-6)*h, 2*w))            
        else:
            i = 12 - l
            if i < 6:
                SliderOverview.paste(Image.fromarray(ambient), (i*h, w))
            else:
                SliderOverview.paste(Image.fromarray(ambient), ((i-6)*h, 2*w))  
        stop = datetime.now() 
        
    
        print("Results create time: " +str(stop-start))
        
        chamberstop = datetime.now()
        print("Chambertime: " +str(chamberstop-chamberstart))
        
    SliderOverview.save(r'/home/sqi/SQI_results/Slider%s_overview.jpg' %(n))
    sliderstop = datetime.now()
    print("Slidertime: " +str(sliderstop-sliderstart))
    
    #SliderOverview.show()
    #cv2.waitKey(0)
display = Image.fromarray(display)
display.save(r'/home/sqi/SQI_results/display%s.jpg' %(datetime.now()))
"""
for n in range(0, int(test_size)):
    SliderOverview.save(r'/home/sqi/SQI_results/Slider%s_overview.jpg' %(n))
print("images saved)
"""

Run_stop = datetime.now()
print("Startup time: " +str(Startup_stop-Startup_start))
print("Run time: " +str(Run_stop-Run_start))

homing()
camera.StopGrabbing()
camera.Close()
time.sleep(0.2)
LIGHT('BLUE_LIGHT_OFF')
time.sleep(0.2)
LIGHT('LASER_LIGHT_OFF')
time.sleep(0.2)
LIGHT('AMBIENT_LIGHT_OFF')
GPIO.output(X_en, GPIO.HIGH)
GPIO.output(Y_en, GPIO.LOW)
GPIO.output(Z_en, GPIO.HIGH)


    
