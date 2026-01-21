#=============================================================================
# Copyright 2017 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ('Confidential Information'). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
#=============================================================================

import PyCapture2
from time import sleep

num = 0

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print('PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3]))
    print()

def print_camera_info(cam):
    cam_info = cam.getCameraInfo()
    print('\n*** CAMERA INFORMATION ***\n')
    print('Serial number - %d' % cam_info.serialNumber)
    print('Camera model - %s' % cam_info.modelName)
    print('Camera vendor - %s' % cam_info.vendorName)
    print('Sensor - %s' % cam_info.sensorInfo)
    print('Resolution - %s' % cam_info.sensorResolution)
    print('Firmware version - %s' % cam_info.firmwareVersion)
    print('Firmware build time - %s' % cam_info.firmwareBuildTime)
    print()

def simple_callback(data, *args):
    global num
    num += 1
    print('\nReceived callback for event: {}'.format(data.eventID))
    print('User data: {}'.format(', '.join(map(str, args))))
    print('Num is now {}'.format(num))
    print()
    
def enable_embedded_timestamp(cam, enable_timestamp):
    embedded_info = cam.getEmbeddedImageInfo()
    if embedded_info.available.timestamp:
        cam.setEmbeddedImageInfo(timestamp = enable_timestamp)
        if enable_timestamp:
            print('\nTimeStamp is enabled.\n')
        else:
            print('\nTimeStamp is disabled.\n') 

def grab_images(cam, num_images_to_grab):
    cam.registerEvent('EventExposureEnd'.encode('utf-8'), simple_callback, 1, 'a', 3.1415)
    prev_ts = None
    framerate = cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE).absValue
    
    for i in range(num_images_to_grab):
        image = cam.retrieveBuffer()
        ts = image.getTimeStamp()
        if prev_ts:
            diff = (ts.cycleSeconds - prev_ts.cycleSeconds) * 8000 + (ts.cycleCount - prev_ts.cycleCount)
            print('Timestamp [ %d %d ] - %d' % (ts.cycleSeconds, ts.cycleCount, diff))
        sleep(1.0/framerate)
        prev_ts = ts
    
    cam.deregisterAllEvents()
    
    newimg = image.convert(PyCapture2.PIXEL_FORMAT.BGR)
    print('Saving the last image to fc2TestImage.png')
    newimg.save('fc2EventExImage.png'.encode('utf-8'), PyCapture2.IMAGE_FILE_FORMAT.PNG)

#
# Example Main
#

# Print PyCapture2 Library Information
print_build_info()

# Ensure sufficient cameras are found
bus = PyCapture2.BusManager()
num_cams = bus.getNumOfCameras()
print('Number of cameras detected: ', num_cams)
if not num_cams:
    print('Insufficient number of cameras. Exiting...')
    exit()

# Select camera on 0th index
c = PyCapture2.Camera()
uid = bus.getCameraFromIndex(0)
c.connect(uid)

# Print camera details
print_camera_info(c)

# Enable camera embedded timestamp
enable_embedded_timestamp(c, True)

print('Starting image capture...')
c.startCapture()
grab_images(c, 100)
c.stopCapture()

# Disable camera embedded timestamp
enable_embedded_timestamp(c, False)
c.disconnect()

input('Done! Press Enter to exit...\n')