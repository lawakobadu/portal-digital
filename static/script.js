let video = document.getElementById("video")

const startStream = () => {
    navigator.mediaDevices.getUserMedia({
        video : {
            width: 600,
            height: 450
        },
        audio : false
    }).then((stream) => {
        video.srcObject = stream
    })
}

startStream()