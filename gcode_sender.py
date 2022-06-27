import time
import serial
s = 0

def openn(com):
	global s
	# inincjowanie otwarcia portu COM
	s = serial.Serial(com,250000)
	print ('Opening Serial Port')
	s.write(str.encode("\r\n\r\n")) # Wysłanie kilku znaków 
	time.sleep(2)   
	s.flushInput()  # czyszczenie bufora
	return

#### Wysyłanie całego pliku GCODE do plotera
def send(src):
	global s
	# Otwarcie pliku
	f = open(src,'r');
	# print ('Opening gcode file')
	for line in f:
		# l = removeComment(line)
		l = line.strip() # rozdzielenie linijki do znaku EOL
		if  (l.isspace()==False and len(l)>0) :
			# print ('Sending: ' + l)
			s.write(str.encode(l + '\n')) # Wysłanie linijki kodu
			grbl_out = s.readline() # Odpowiedź plotera
			# print ( grbl_out.strip())
		time.sleep(0.1)
	return

#### Wysyłanie jedniej linijki GCODE do plotera
def send_line(ls):
	global s
	s.write(str.encode("\r\n\r\n")) 
	time.sleep(0.5)   
	s.flushInput() 
	s.write(str.encode(ls + '\n')) 
	grbl_out = s.readline() 
	# print ( grbl_out.strip())

def close():
	global s
	# Zamknięcie pliku i portu COM
	f.close()
	s.close()
	return

# test zbazowania
def baza():
	global s
	s.write(str.encode("M119" + '\n')) # Send g-code block
	grbl_out = s.readline() # Wait for response with carriage return
	print ( grbl_out.strip())
	return grbl_out.strip()
