console.log(faceapi)

let faceMatcher = null;
let intervalId = null;

const run = async () => {
    // loading the models is going to use await
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    const videoFeedEl = document.getElementById('video-feed');
    videoFeedEl.srcObject = stream;

    // we need to load our models
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
    ]);

    // make the canvas the same size and in the same location as the video feed
    const canvas = document.getElementById('canvas');
    canvas.style.left = videoFeedEl.offsetLeft + 'px';
    canvas.style.top = videoFeedEl.offsetTop + 'px';
    canvas.width = videoFeedEl.width;
    canvas.height = videoFeedEl.height;

    // Event listener for image upload
    const imageUpload = document.getElementById('imageUpload');
    const imageInfo = document.getElementById('imageInfo');
    const loadingText = document.getElementById('loadingText');

    imageUpload.addEventListener('change', async () => {
        const file = imageUpload.files[0];
        if (file) {
            // Clear any existing interval
            if (intervalId) {
                clearInterval(intervalId);
            }

            // Display loading text
            loadingText.style.display = 'block';

            setTimeout(() => {
                loadingText.style.display = 'none';
            }, 2000);

            imageInfo.textContent = `Image Uploaded: ${file.name}`;

            // Convert the file to an image
            const refFace = await faceapi.bufferToImage(file);
            const refFaceAiData = await faceapi.detectAllFaces(refFace).withFaceLandmarks().withFaceDescriptors().withAgeAndGender();
            faceMatcher = new faceapi.FaceMatcher(refFaceAiData);

            // Restart facial detection with points
            intervalId = setInterval(async () => {
                // Get the video feed and give to detectAllFaces method
                let faceAIData = await faceapi.detectAllFaces(videoFeedEl).withFaceLandmarks().withFaceDescriptors().withAgeAndGender();

                // Clear the canvas
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

                // Resize and draw detections and landmarks
                faceAIData = faceapi.resizeResults(faceAIData, videoFeedEl);
                faceapi.draw.drawDetections(canvas, faceAIData);
                faceapi.draw.drawFaceLandmarks(canvas, faceAIData);

                // Guess age and gender 
                faceAIData.forEach(face => {
                    const { age, gender, genderProbability, detection, descriptor } = face;
                    const genderText = `${gender}`;
                    const ageText = `${Math.round(age)} years`;
                    const textField = new faceapi.draw.DrawTextField([genderText, ageText], detection.box.topRight);
                    textField.draw(canvas);

                    // Check the face match
                    let label = faceMatcher.findBestMatch(descriptor).toString();
                    let options;
                    let boxColor;

                    if (label.includes("unknown")) {
                        options = { label: "No Match" };
                        // Red box for no match
                        boxColor = 'red'; 
                    } else {
                        options = { label: "Match" };
                        // Green box for match
                        boxColor = 'green'; 
                    }

                    // Draw bounding box with color based on match
                    const drawBox = new faceapi.draw.DrawBox(detection.box, { ...options, boxColor });
                    drawBox.draw(canvas);

                });

            }, 50);
        } else {
            imageInfo.textContent = '';
        }
    });
};

run();