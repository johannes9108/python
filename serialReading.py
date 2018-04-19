import serial,time,io

ser = serial.Serial()

ser.baudrate = 115200
ser.port =  'COM7'
ser.timeout = 0
ser.parity = 'N'
ser.stopbits = 1
ser.bytesize = 8
run = True
ser.open()

while(run == True):
    s = ser.readline()
    text = str(s)
    if(text.__len__()>3):
        print(text)
    time.sleep(0.4)
ser.close()