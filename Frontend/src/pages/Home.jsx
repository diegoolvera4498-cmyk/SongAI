import { useState } from "react";

import Recorder from "../components/Recorder";
import ResultCard from "../components/ResultCard";

export default function Home() {

    const [result, setResult] = useState(null);

    return (
        <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            marginTop: "50px",
            fontFamily: "Arial"
        }}>

            <h1>SONGAI 🎵</h1>

            <p>
                Analizador inteligente de voz
            </p>

            <Recorder setResult={setResult} />

            <ResultCard result={result} />

        </div>
    );
}