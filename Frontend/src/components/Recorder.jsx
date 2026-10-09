import { useState, useRef } from "react";
import { sendAudio } from "../services/api";

export default function Recorder({ setResult }) {

    const [recording, setRecording] = useState(false);

    const mediaRecorder = useRef(null);

    const audioChunks = useRef([]);

    const startRecording = async () => {

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true,
        });

        mediaRecorder.current = new MediaRecorder(stream);

        audioChunks.current = [];

        mediaRecorder.current.start();

        setRecording(true);

        mediaRecorder.current.ondataavailable = (event) => {
            audioChunks.current.push(event.data);
        };
    };


    const stopRecording = async () => {

        mediaRecorder.current.stop();

        mediaRecorder.current.onstop = async () => {

            const audioBlob = new Blob(audioChunks.current, {
                type: "audio/webm",
            });

            console.log("Enviando audio...");

            const result = await sendAudio(audioBlob);

            console.log(result);

            setResult(result);
        };

        setRecording(false);
    };

    return (
        <div>

            {!recording ? (
                <button onClick={startRecording}>
                    🎙️ Grabar
                </button>
            ) : (
                <button onClick={stopRecording}>
                    ⏹️ Detener
                </button>
            )}

        </div>
    );
}