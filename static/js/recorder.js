let isRecording = false;
let mediaRecorder;
let audioChunks = [];
const recordButton = document.getElementById("recordButton");
const recordingStatus = document.getElementById("recordingStatus");
const assistanceResponse = document.getElementById("assistanceResponse");
const modelId = document.getElementById("modelId");
const userInputText = document.getElementById("userInputText");
const modelResponseText = document.getElementById("modelResponseText");
const modelResponsePlayer = document.getElementById("modelResponsePlayer");

recordButton.addEventListener("click", async () => {
  assistanceResponse.style.display = "none";
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  // Request microphone access
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  // Record data in chunks
  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
  };

  // Handle stop recording event
  mediaRecorder.onstop = () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    sendAudio(audioBlob);
  };

  // Start recording
  mediaRecorder.start();
  isRecording = true;
  recordingStatus.textContent = "Status: Recording...";
}

function stopRecording() {
  // Stop the recording
  mediaRecorder.stop();
  isRecording = false;
  recordingStatus.textContent = "Status: Processing...";
}

async function sendAudio(audioBlob) {
  // Send the audio to the server
  const formData = new FormData();
  formData.append("audio_data", audioBlob, "recording.wav");

  const response = await fetch("/process-audio", {
    method: "POST",
    body: formData,
  });

  const jsonResponse = await response.json();
  console.log({ jsonResponse });
  recordingStatus.textContent = "Status: Idle";
  modelId.innerHTML = jsonResponse.model_id;
  userInputText.innerHTML = jsonResponse.user_input;
  console.log("jsonResponse.response_text....", jsonResponse.response_text);
  console.log("marked typeof", typeof marked);
  console.log(
    "marked jsonResponse.response_text",
    marked.parse(jsonResponse.response_text)
  );
  modelResponseText.innerHTML = marked.parse(jsonResponse.response_text);

  modelResponsePlayer.src = jsonResponse.audio_file_path;
  modelResponsePlayer.load();

  assistanceResponse.style.display = "block";
}
