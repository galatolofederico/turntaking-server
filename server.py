import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from src.analyzer import Analyzer

ROOT = os.path.join(os.path.dirname(__file__), "assets")
logger = logging.getLogger("pc")
pcs = set()
relay_audio = MediaRelay()
MQTTClient = None

class AudioTrackProcessing(MediaStreamTrack):
    """
    Audio stream track that processess AudioFrames from tracks.
    """

    def __init__(self, track:MediaStreamTrack, analyzer:Analyzer):
        super().__init__()
        self.track = track
        self.analyzer = analyzer

    async def recv(self):
        frame = await self.track.recv()
        self.analyzer.analyze(frame)
        return frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    recorder = MediaBlackhole()
    analyzer = Analyzer(
        mqtt_client=MQTTClient,
        mqtt_topic=args.mqtt_topic
    )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            MQTTClient.publish(args.mqtt_topic, "start")
            print("Started Listening")
            relayed_audio = relay_audio.subscribe(track)
            recorder.addTrack(AudioTrackProcessing(relayed_audio, analyzer))
        
        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            MQTTClient.publish(args.mqtt_topic, "stop")
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer) # type: ignore

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC audio / video / data-channels demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server (default: 8080)")
    parser.add_argument("--mqtt-host", default="", help="Host for MQTT server")
    parser.add_argument("--mqtt-port", type=int, default=-1, help="Port for MQTT server")
    parser.add_argument("--mqtt-transport", type=str, default="websockets", help="MQTT transport protocol")
    parser.add_argument("--mqtt-ssl", action="store_true", help="Username for MQTT server")
    parser.add_argument("--mqtt-user", default="", help="Username for MQTT server")
    parser.add_argument("--mqtt-password", default="", help="Password for MQTT server")
    parser.add_argument("--mqtt-topic", default="speaking/status", help="Topic to publish audio data")

    args = parser.parse_args()

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    if len(args.mqtt_host) > 0 and args.mqtt_port > 0:
        import paho.mqtt.client as mqtt
        MQTTClient = mqtt.Client(transport=args.mqtt_transport)
        if args.mqtt_ssl:
            MQTTClient.tls_set()
        if len(args.mqtt_user) > 0:
            MQTTClient.username_pw_set(username=args.mqtt_user , password=args.mqtt_password)

        def on_connect(client, userdata, flags, rc):
            print("MQTT: Connected with result code "+str(rc))

        MQTTClient.on_connect = on_connect

        MQTTClient.__topic = args.mqtt_topic
        MQTTClient.connect(args.mqtt_host, args.mqtt_port, 60)
        MQTTClient.loop_start()


    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
