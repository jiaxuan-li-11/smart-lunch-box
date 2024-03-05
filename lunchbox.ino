#include <ESP8266WiFi.h>
#include <ArduinoJson.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <SoftwareSerial.h>
#include <string>
#include <Wire.h>
#include <ArduCAM.h>
#include <SPI.h>
#include <HX711_ADC.h>
#include <EEPROM.h>

// HX711 setting
//pins:
const int HX711_dout = 0;  //set HX711 dout pin
const int HX711_sck = 2;   //set HX711 sck pin
//HX711 constructor:
HX711_ADC LoadCell(HX711_dout, HX711_sck);

const int calVal_eepromAdress = 0;
unsigned long t = 0;
//set some parameters for future load-cell determination loop
int pa1=1;
int pa2=1;
int m=0;
float stor1[9000];
float stor2[1100];

//arduCAM setting
// pins:
const int CS = 16;
ArduCAM myCAM(OV2640, CS);

//network setting
String url = "http://3.86.239.158:8080/weight";
const char *ssid = "Columbia University";
const char *password = "";
const char *host = "3.86.239.158";
int port = 8080;

void setup(){
  Serial.begin(57600);//set to transmit at 57600 bits per second so that they can be monitered
  delay(10);

  // loadcell
  LoadCell.begin();
    // Set calibration for load cell
  float calibrationValue;
  calibrationValue = 696.0;
    //fetch the calibration value from eeprom using ESP8266
  EEPROM.begin(512);  
  EEPROM.get(calVal_eepromAdress, calibrationValue);

  unsigned long stabilizingtime = 2000;  //add a few seconds of stabilizing time to improve the preciscion
  //perform the tare  
  boolean _tare = true;                  
  LoadCell.start(stabilizingtime, _tare);
  if (LoadCell.getTareTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 wiring and pin designations");
    while (1);
  } else {
    LoadCell.setCalFactor(calibrationValue);  // set calibration value (float)
  }

  // arduCAM
  // set the parameters and format
  uint8_t vid, pid;
  uint8_t temp;
  #if defined(__SAM3X8E__)
    Wire1.begin();
  #else
    Wire.begin();
  #endif
  Serial.println("ArduCAM Start!");//send a message to shown ardyCAM set well
  pinMode(CS, OUTPUT);
  //Setting the SPI-related parameters
  SPI.begin();
  SPI.setFrequency(4000000);
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  temp = myCAM.read_reg(ARDUCHIP_TEST1);
  // If the setting fails, send a message to the serial
  if (temp != 0x55){
    Serial.println("SPI1 interface Error!");
    while(1);
  }

  myCAM .wrSensorReg8_8(0xff,0x01);
  myCAM .rdSensorReg8_8(OV2640_CHIPID_HIGH,&vid);
  myCAM .rdSensorReg8_8(OV2640_CHIPID_LOW,&pid);
  if((vid !=0x26)&&((pid != 0x41)||( pid != 0x42)))
    Serial.println("Can't find OV2640 module!");
  else
    Serial.println("OV2640 detected.");

  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  // OV2640_320x240
  // OV2640_640x480   
  // OV2640_800x600   
  // OV2640_1024x768   
  // OV2640_1280x1024  
  // OV2640_1600x1200  
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);
  myCAM.clear_fifo_flag();  

  //Connect with WIFI
  WiFi.begin(ssid,password);
  while(WiFi.status()!=WL_CONNECTED)
  {
    Serial.print(".");
    delay(500);
  }
  Serial.println("");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("WiFi Connected Successfully !");
}


void loop(){

  HTTPClient http;// perform HTTP requests
  WiFiClient client;//creates a client that can connect to to a specified internet as we set before
  Serial.printf("Connecting to %s ...", host);
  //LOAD-CELL
  static boolean newDataReady = 0;
  const int serialPrintInterval = 0; 

  if (LoadCell.update()) newDataReady = true;

  // get smoothed value from the dataset
  if (newDataReady) {
    if (millis() > t + serialPrintInterval) {
      float i = LoadCell.getData();
      Serial.println(i);
  //Determining if the weight on the load cell has increased
      if (i-stor1[pa1-1]>0.2){ m=1;}
  
      if (m==1){
        stor2[pa2]=i;
        //Determine if the value on the load cell is no longer changing 
        if (stor2[pa2]==stor2[pa2-1]){
          //The value read at this point is the most accurate and up-to-date weight value
          m=0;
          //convert the weight value to string type
          String i_str = "";
          i_str += stor2[pa2];
          Serial.println("get new stuff ");//a notification
          //Post the new weight to EC2
          if (WiFi.status() == WL_CONNECTED) {
            http.begin(client, url);
            int httpCode = http.POST(i_str);
            http.end();
          }
          //Take an picture and transfer it to EC2
          camCapture(myCAM, client);
        }
        pa2+=1;
      }
      stor1[pa1]=i; 
      pa1 +=1;
      newDataReady = 0;
      t = millis();
      delay(250);
    }
  }

  //If want to tare, input 't' to the serial
  if (Serial.available() > 0) {
    char inByte = Serial.read();
    if (inByte == 't') LoadCell.tareNoDelay();
  }

  //Check if last tare operation is complete
  if (LoadCell.getTareStatus() == true) {
    Serial.println("Tare complete");
  }
}

//Fuction for camera using
void camCapture(ArduCAM myCAM, WiFiClient client){
  //Setting up arduCAM's data buffer
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)); 
  Serial.println("CAM Capturing");
  size_t len = myCAM.read_fifo_length();
  //Determine whether the picture is right
  if (len >= 0x07ffff){
    myCAM.clear_fifo_flag();
    Serial.println("Over size.");
    return;
  }else if (len == 0 ){
    myCAM.clear_fifo_flag();
    Serial.println("Size is 0.");
    return;
  }
  //Write the picture into the buffer
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  #if !(defined (ARDUCAM_SHIELD_V2) && defined (OV2640_CAM))
    SPI.transfer(0xFF);
  #endif
  //Post the picture to EC2
  if(client.connect(host, port)){
    String postRequest = "POST /capture HTTP/1.1\r\n";
    postRequest += "Host:13.58.104.6:8080\r\n";
    postRequest += "Content-Type: image/jpeg\r\n";
    postRequest += "Content-Length: " + String(len) + "\r\n\r\n";
    client.print(postRequest);
    static const size_t bufferSize = 4096;
    static uint8_t buffer[bufferSize] = {0xFF}; 
    while (len) {
      size_t will_copy = (len < bufferSize) ? len : bufferSize;
      SPI.transferBytes(&buffer[0], &buffer[0], will_copy);
      if (!client.connected()) break;
      client.write(&buffer[0], will_copy);
      len -= will_copy;
      Serial.print("Length: ");
      Serial.println(will_copy);
    }
    client.print("\r\n--boundary--\r\n");
    myCAM.CS_HIGH();
    while(client.connected() || client.available()){
      if (client.available()){
        String line = client.readStringUntil('\n');
        Serial.println(line);
      }
    }
    client.stop();
    Serial.println("Disconnected");
  }
  else{
    Serial.println("connection failed!]");
    client.stop();
  }

}






































