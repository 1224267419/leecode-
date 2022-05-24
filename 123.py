import serial

ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM3'
ser.open()
print(ser.isOpen())
ser.write('1'.encode())