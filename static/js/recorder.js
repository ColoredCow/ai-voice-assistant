let isRecording = false;
let mediaRecorder;
let audioChunks = [];
const recordButton = document.getElementById("recordButton");
const audioUpload = document.getElementById("audioUpload");
const recordingStatus = document.getElementById("recordingStatus");
const assistanceResponse = document.getElementById("assistanceResponse");
const modelId = document.getElementById("modelId");
const userInputText = document.getElementById("userInputText");
const modelResponseText = document.getElementById("modelResponseText");
const modelResponsePlayer = document.getElementById("modelResponsePlayer");
const modelRequestPlayer = document.getElementById("modelRequestPlayer");

recordButton.addEventListener("click", async () => {
  assistanceResponse.style.display = "none";
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

// Handle uploaded audio file
audioUpload.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) {
    recordingStatus.textContent = "Status: Processing...";
    await sendAudio(file);
  }
});

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    sendAudio(audioBlob);
  };

  mediaRecorder.start();
  isRecording = true;
  recordingStatus.textContent = "Status: Recording...";
}

function stopRecording() {
  mediaRecorder.stop();
  isRecording = false;
  recordingStatus.textContent = "Status: Processing...";
}

async function sendAudio(audioBlobOrFile) {
  const formData = new FormData();
  formData.append("audio_data", audioBlobOrFile, "recording.wav");

  // Send audio and get metadata response
  const response = await fetch("/process-audio", {
    method: "POST",
    body: formData,
  });
  const jsonResponse = await response.json();

  // Display metadata
  recordingStatus.textContent = "Status: Idle";
  modelId.innerHTML = jsonResponse.model_id;
  userInputText.innerHTML = jsonResponse.user_input;
  modelRequestPlayer.src = jsonResponse.recorded_audio_path;
  modelRequestPlayer.load();
  assistanceResponse.style.display = "block";

  // Start streaming the chatbot response
  const eventSource = new EventSource(
    `/stream-response?user_input=${encodeURIComponent(jsonResponse.user_input)}`
  );

  eventSource.onmessage = function (event) {
    const chunk = event.data;
    const lastChar = modelResponseText.innerHTML.slice(-1);

    if (chunk === "[END OF RESPONSE]") {
      // End of stream reached, close the connection and trigger next step
      console.log("closing the stream");
      eventSource.close();
      convertToAudio();
      return true;
    }

    // Check if the chunk starts with a space and add it only if needed
    if (lastChar !== " " && chunk[0] !== " " && chunk.length > 0) {
      modelResponseText.innerHTML += " ";
    }

    modelResponseText.innerHTML += chunk;
  };

  // Trigger the next request once the streaming completes
  eventSource.onclose = function () {
    console.log("Streaming completed. Sending next request...");
    // Call the next function or send another request after streaming completes
    convertToAudio(); // Replace with your next function or request
  };

  eventSource.onerror = function (event) {
    console.error("Error in EventSource:", event);
    eventSource.close();
  };
}

async function convertToAudio() {
  let text = modelResponseText.innerHTML;
  // URL-encode the text to handle special characters
  const encodedText = encodeURIComponent(text);

  // Send the encoded text in the request
  const response = await fetch(`/process-tts?text=${encodedText}`);

  // Handle the response
  const jsonResponse = await response.json();
  modelResponsePlayer.src = jsonResponse.audio_file_path;
  modelResponsePlayer.load();
}
