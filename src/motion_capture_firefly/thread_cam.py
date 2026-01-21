import PyCapture2
import multiprocessing
import time

# Register Constants
SOFTWARE_TRIGGER = 0x62C
FIRE_VALUE = 0x80000000  # Bit 31 set

def poll_for_trigger_ready(cam):
    """
    Polls the register until Bit 31 is 0, meaning the camera 
    is ready to accept a new software trigger.
    """
    while True:
        # readRegister returns the 32-bit value of the register
        reg_val = cam.readRegister(SOFTWARE_TRIGGER)
        # If Bit 31 is 0, the camera is ready
        if not (reg_val & FIRE_VALUE):
            break
        time.sleep(0.001) # Small sleep to prevent CPU hogging

def enable_embedded_timestamp(cam, enable_timestamp):
    embedded_info = cam.getEmbeddedImageInfo()
    if embedded_info.available.timestamp:
        cam.setEmbeddedImageInfo(timestamp = enable_timestamp)
        if enable_timestamp :
            print('\nTimeStamp is enabled.\n')
        else:
            print('\nTimeStamp is disabled.\n')


class CameraWorker:
    def __init__(self, cam_index, barrier, stop_event):
        self.cam_index = cam_index
        self.barrier = barrier
        self.stop_event = stop_event

    def run(self):
        bus = PyCapture2.BusManager()
        cam = PyCapture2.Camera()
        enable_embedded_timestamp(cam, True)

        try:
            cam.connect(bus.getCameraFromIndex(self.cam_index))
            
            # Configure Software Trigger
            trigger_mode = cam.getTriggerMode()
            trigger_mode.onOff = True
            trigger_mode.source = 7  # Software
            cam.setTriggerMode(trigger_mode)
            cam.startCapture()

            print(f"[Cam {self.cam_index}] Standing by...")

            while not self.stop_event.is_set():
                # 1. Wait for all cameras to be physically ready
                poll_for_trigger_ready(cam)
                
                # 2. Synchronize all processes at the "Starting Line"
                # This ensures no camera triggers while another is still polling
                try:
                    self.barrier.wait(timeout=5)
                except multiprocessing.TimeoutError:
                    continue

                # 3. FIRE TRIGGER
                cam.writeRegister(SOFTWARE_TRIGGER, FIRE_VALUE)
                
                # 4. Retrieve Image
                try:
                    image = cam.retrieveBuffer()
                    # Success logic here
                    print(f"[Cam {self.cam_index}] Captured image with timestamp: {image.getTimeStamp()}")
                except PyCapture2.Fc2error as e:
                    print(f"[Cam {self.cam_index}] Capture Error: {e}")

        finally:
            cam.stopCapture()
            cam.disconnect()

class CameraManager:
    def __init__(self):
        self.bus = PyCapture2.BusManager()
        self.num_cams = self.bus.getNumOfCameras()
        # Barrier ensures exactly N processes reach the same line before any proceed
        self.barrier = multiprocessing.Barrier(self.num_cams)
        self.stop_event = multiprocessing.Event()
        self.processes = []

    def start(self):
        for i in range(self.num_cams):
            worker = CameraWorker(i, self.barrier, self.stop_event)
            p = multiprocessing.Process(target=worker.run)
            p.start()
            self.processes.append(p)

    def stop(self):
        self.stop_event.set()
        # Abort barrier to release any processes stuck waiting
        self.barrier.abort()
        for p in self.processes:
            p.join()

if __name__ == '__main__':
    mgr = CameraManager()
    mgr.start()
    try:
        # The cameras will now loop synchronously as fast as the 
        # slowest camera/processing allows.
        while True: time.sleep(1)
    except KeyboardInterrupt:
        mgr.stop()
