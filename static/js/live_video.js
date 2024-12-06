document.addEventListener("DOMContentLoaded", function () {
    let liveStream;
    let videoElement = document.getElementById("liveVideo");
    let intervalId;
    let frameCounter = 0; // To keep track of frames (e.g., frame_1, frame_2, ...)

    // Start Live Matching (Webcam Access)
    document.getElementById("startLive").addEventListener("click", function () {
        const files = document.getElementById("samples").files;
        if (!files.length) {
            alert("Please upload sample images first");
            return;
        }

        const formData = new FormData();
        formData.append("query_image", files[0]);

        // Access the webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                liveStream = stream;
                videoElement.srcObject = stream;
                document.getElementById("startLive").disabled = true;
                document.getElementById("stopLive").disabled = false;
                document.getElementById("liveStatus").innerText = "Live Matching: On";

                // Start capturing frames every 1 second (adjust as needed)
                intervalId = setInterval(() => {
                    captureFrameAndProcess();
                }, 1000);
            })
            .catch(function (error) {
                alert("Error accessing the webcam: " + error.message);
            });
    });

    // Function to capture a frame from the video and send it to the backend for processing
    function captureFrameAndProcess() {
        let canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvas.getContext("2d").drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
        // Convert the canvas image to a blob or base64 string
        canvas.toBlob((blob) => {
            console.log("Captured frame as blob:", blob);  // Add this log to verify the frame
            const formData = new FormData();
            formData.append("frame", blob, `frame_${frameCounter++}.jpg`);  // Incrementing frame ID
    
            // Send the frame to the backend for processing
            fetch("/process_live", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log(data); // Process the response data
                updateLiveResults(data);
            })
            .catch(error => {
                console.error(error);
                document.getElementById("liveStatusMessage").innerHTML = `<div class="alert alert-danger">Error processing video.</div>`;
            });
        }, "image/jpeg");
    }
    
    // Update the UI with the live matching results
    function updateLiveResults(data) {
        if (data.results && data.results.length > 0) {
            // Process and display results here
            document.getElementById("liveStatusMessage").innerHTML = `
                <div class="alert alert-info">
                    Video ID: ${data.video_id || 'N/A'} <br>
                    Mismatches: ${data.results.map(result => `Frame: ${result.frame}, Description: ${result.description}`).join('<br>')}
                </div>
            `;

            // Update the report table with frame-specific data
            let reportTableBody = document.querySelector("#reportDetails tbody");
            data.results.forEach(result => {
                let row = document.createElement("tr");
                row.innerHTML = `
                    <td>${result.frame || 'N/A'}</td>
                    <td>Similarity Score: ${result.similarity || 'N/A'}</td>
                `;
                reportTableBody.appendChild(row);
            });
        } else {
            // If no results, display an appropriate message
            document.getElementById("liveStatusMessage").innerHTML = "<div class='alert alert-warning'>No matching frames found.</div>";
        }
    }

    // Stop Live Matching (Stop Webcam Access)
    document.getElementById("stopLive").addEventListener("click", function () {
        if (liveStream) {
            let tracks = liveStream.getTracks();
            tracks.forEach(track => track.stop());  // Stop the video stream
            videoElement.srcObject = null;  // Remove video source
        }
        document.getElementById("startLive").disabled = false;
        document.getElementById("stopLive").disabled = true;
        document.getElementById("liveStatus").innerText = "Live Matching: Off";
        document.getElementById("liveStatusMessage").innerHTML = "<div class='alert alert-warning'>Live Matching Stopped.</div>";

        // Clear the interval when live matching is stopped
        clearInterval(intervalId);
    });
});
