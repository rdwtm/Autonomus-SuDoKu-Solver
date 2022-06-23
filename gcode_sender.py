
import time
import serial

def removeComment(string):
	if (string.find(';')==-1):
		return string
	else:
		return string[:string.index(';')]
 
def open():
	# Open serial port
	s = serial.Serial("COM3",250000)
	print ('Opening Serial Port')
	return
def send(src):
	# Open g-code file
	sc=r"C:\Users\raven\OneDrive\Pulpit\gkody\ou_0011.ngc"
	f = open(sc,'r');
	print ('Opening gcode file')
	
	# Wake up 
	s.write(str.encode("\r\n\r\n")) # Hit enter a few times to wake the Printrbot
	time.sleep(2)   # Wait for Printrbot to initialize
	s.flushInput()  # Flush startup text in serial input
	print ('Sending gcode')
	
	# Stream g-code
	for line in f:
		l = removeComment(line)
		l = l.strip() # Strip all EOL characters for streaming
		if  (l.isspace()==False and len(l)>0) :
			print ('Sending: ' + l)
			s.write(str.encode(l + '\n')) # Send g-code block
			grbl_out = s.readline() # Wait for response with carriage return
			print ( grbl_out.strip())
	
	# Wait here until printing is finished to close serial port and file.
	raw_input("  Press <Enter> to exit.")
	return


def close():
	# Close file and serial port
	f.close()
	s.close()
	return