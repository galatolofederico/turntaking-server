import time
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("/test/ciao")

def on_message(client, userdata, msg):
    print("Received message:", msg.payload.decode())

client = mqtt.Client(transport="websockets")
client.tls_set()
client.username_pw_set(username="NotMyRealToken",password="")

client.on_connect = on_connect
client.on_message = on_message

client.connect("mqtt.galatolo.xyz", 443, 60)
client.loop_start()
while True:
    client.publish("/test/topic", "ciao")
    time.sleep(1)

client.disconnect()