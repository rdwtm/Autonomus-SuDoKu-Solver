# Autonomus-SuDoKu-Solver
 Project of device witch solve SuDoKu puzzle and write solution on paper card, where puzzle is.


This is my project of autonomous SuDoKu solver. 
Construction is based on Prusa r3 Steel, but work area was increased tripple. 
Image is transmitting by camera server from my phone, which was placed at the top of construction.
To detect numbers i use my own trained OCR model. 
Solver is based on BackTraking search algorithm.
Computer is linked to manipulator by UART. 

The device works as follows:
User launch the device. Manipulator goes to base position. Then the table is moving forward to user. 
Then user put and stick papercard with sudoku puzzle. 
After that click enter key.
Device do photo of puzzle. OCR recognize digits and solver find solution. 
Plotter write solution by pen attached to carriage
