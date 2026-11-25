"""
Simulate a G.711 VoIP call by sending RTP-formatted UDP packets.
RTP header (12 bytes) + 160 bytes payload = 172 bytes/packet at 50 pkt/s
This matches the ISCXVPN2016 VoIP traffic pattern.
"""
import socket, struct, random, time, os

# RTP parameters
PAYLOAD_TYPE = 0       # PCMU / G.711
SSRC = random.randint(0, 0xFFFFFFFF)
SEQ = random.randint(0, 65535)
TIMESTAMP = random.randint(0, 0xFFFFFFFF)
PAYLOAD_SIZE = 160     # 20ms of G.711 @ 8kHz
INTERVAL = 0.020       # 20ms between packets = 50 pkt/s

# Send to router (packets go out en0, tshark captures them)
DST_IP = "10.0.16.1"
DST_PORT = 5004

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending RTP stream to {DST_IP}:{DST_PORT} at 50 pkt/s ...")
sent = 0
start = time.time()

while time.time() - start < 100:  # 100 seconds max
    # Build RTP header
    header = struct.pack('!BBHII',
        0x80,           # V=2, P=0, X=0, CC=0
        PAYLOAD_TYPE,   # M=0, PT=0 (PCMU)
        SEQ & 0xFFFF,
        TIMESTAMP & 0xFFFFFFFF,
        SSRC
    )
    payload = bytes([random.randint(0, 255) for _ in range(PAYLOAD_SIZE)])
    sock.sendto(header + payload, (DST_IP, DST_PORT))
    
    SEQ += 1
    TIMESTAMP += 160  # 160 samples per 20ms frame
    sent += 1
    time.sleep(INTERVAL)

sock.close()
print(f"Sent {sent} RTP packets")
