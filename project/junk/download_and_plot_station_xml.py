from obspy.clients.fdsn import Client
from obspy import UTCDateTime

import os

# Define paths and parameters
path = os.getcwd()
dst = os.path.join(path, 'Stationxml/')
os.makedirs(dst, exist_ok=True)

client = Client("IRIS")
netsclts = {"II": "IRIS"}
starttime = UTCDateTime(2001, 1, 1, 0, 0, 0)
endtime = UTCDateTime(2020, 12, 12, 23, 59, 59)

# Loop through networks and retrieve metadata
for net in list(netsclts.keys()):
    client = Client(netsclts[net])
    inventory = client.get_stations(
        starttime=starttime,
        endtime=endtime,
        network=net,
        sta="MBAR",
        loc="00",
        channel="*",
        level="response"
    )
    inventory.write(dst + f"metadata.{net}.xml", format="STATIONXML")
    print(f"Metadata for {net} written to {dst}")

    # Plot instrument response for displacement
    for network in inventory:
        for station in network:
            for channel in station:
                if channel in ["BHZ","BHN","BHE"]:
                    response = channel.response
                    if response:
                        print(f"Plotting instrument response for {channel.code}")
                        response.plot(0.001, output="DISP")  # Frequency 0.001 Hz

