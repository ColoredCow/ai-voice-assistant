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
const submitTextButton = document.getElementById("submitTextButton");
const textInput = document.getElementById("textInput");

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

// Event listener for text input submission
submitTextButton.addEventListener("click", async () => {
  const text = textInput.value;
  if (text) {
    recordingStatus.textContent = "Status: Processing...";
    await sendText(text);
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

// Function to send text input to the server
async function sendText(text) {
  const formData = new FormData();
  formData.append("text_input", text);

  const response = await fetch("/process-text", {
    method: "POST",
    body: formData,
  });

  const jsonResponse = await response.json();
  console.log({ jsonResponse });
  recordingStatus.textContent = "Status: Idle";
  modelId.innerHTML = jsonResponse.model_id;
  userInputText.innerHTML = jsonResponse.user_input;
  modelResponseText.innerHTML = marked.parse(jsonResponse.response_text);

  // Handle model response audio if available
  if (jsonResponse.audio_file_path) {
    modelResponsePlayer.src = jsonResponse.audio_file_path;
    modelResponsePlayer.load();
    modelResponsePlayer.style.display = 'block';
  } else {
    modelResponsePlayer.style.display = 'none';
  }

  assistanceResponse.style.display = "block";
}
