let isRecording = false;
let mediaRecorder;
let audioChunks = [];
const recordButton = document.getElementById("recordButton");
const audioUpload = document.getElementById("audioUpload");
const recordingStatus = document.getElementById("recordingStatus");
const assistanceResponse = document.getElementById("assistanceResponse");
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

  const response = await fetch("/process-audio", {
    method: "POST",
    body: formData,
  });

  const jsonResponse = await response.json();
  console.log({ jsonResponse });
  recordingStatus.textContent = "Status: Idle";
  userInputText.textContent = jsonResponse.user_input;
  modelResponseText.textContent = jsonResponse.response_text;

  modelResponsePlayer.src = jsonResponse.audio_file_path;
  modelResponsePlayer.load();

  assistanceResponse.style.display = "block";
}
