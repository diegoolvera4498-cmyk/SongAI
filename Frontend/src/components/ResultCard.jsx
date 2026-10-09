export default function ResultCard({ result }) {

    if (!result) return null;

    return (

        <div style={{
            marginTop: "20px",
            padding: "20px",
            border: "1px solid gray",
            borderRadius: "10px"
        }}>

            <h2>Resultado del análisis</h2>

            <p>
                <strong>Tipo de voz:</strong>
                {" "}
                {result.voice_type}
            </p>

            <p>
                <strong>Confianza IA:</strong>
                {" "}
                {(result.confidence * 100).toFixed(2)}%
            </p>

            <hr />

            <h3>Probabilidades</h3>

            <p>
                Aguda:
                {" "}
                {(result.probabilities[0] * 100).toFixed(2)}%
            </p>

            <p>
                Grave:
                {" "}
                {(result.probabilities[1] * 100).toFixed(2)}%
            </p>

        </div>
    );
}