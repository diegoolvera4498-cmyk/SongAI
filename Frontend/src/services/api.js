const API_URL = "http://127.0.0.1:5000";

export async function sendAudio(audioBlob) {

    const formData = new FormData();

    formData.append("audio", audioBlob, "recording.webm");

    const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
    });

    return await response.json();
}