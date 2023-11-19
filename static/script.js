let video1 = document.getElementById("video1");
let video2 = document.getElementById("video2");

const startStream = () => {
    navigator.mediaDevices.enumerateDevices()
        .then(devices => {
            const videoDevices = devices.filter(device => device.kind === 'videoinput');

            if (videoDevices.length >= 2) {
                navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: { exact: videoDevices[0].deviceId },
                        width: 600,
                        height: 450
                    },
                    audio: false
                }).then((stream1) => {
                    video1.srcObject = stream1;
                }).catch((error) => {
                    console.error("Error accessing media devices:", error);
                });

                navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: { exact: videoDevices[1].deviceId },
                        width: 600,
                        height: 450
                    },
                    audio: false
                }).then((stream2) => {
                    video2.srcObject = stream2;
                }).catch((error) => {
                    console.error("Error accessing media devices:", error);
                });
            } else {
                console.error("Not enough video devices available.");
            }
        })
        .catch(error => {
            console.error("Error enumerating devices:", error);
        });
};

startStream();
