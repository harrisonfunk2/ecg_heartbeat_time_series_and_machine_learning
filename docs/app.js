const LABELS = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Fusion",
    "Unknown"
];

let session;
let samples = [];

let previousSampleIndex = -1;


// ---------------------------------------
// DOM elements
// ---------------------------------------

const predictionElement =
    document.getElementById("prediction");

const confidenceElement =
    document.getElementById("confidence");

const trueLabelElement =
    document.getElementById("trueLabel");

const statusElement =
    document.getElementById("status");

const probabilityBars =
    document.getElementById("probabilityBars");

const canvas =
    document.getElementById("ecgCanvas");

const ctx =
    canvas.getContext("2d");


// ---------------------------------------
// Load ONNX model + ECG samples
// ---------------------------------------

async function initialize() {

    try {

        statusElement.textContent =
            "Loading ECG model...";

        session =
            await ort.InferenceSession.create(
                "./ecg_cnn.onnx",
                {
                    executionProviders: ["wasm"]
                }
            );

        const response =
            await fetch(
                "./demo_samples.json"
            );

        if (!response.ok) {
            throw new Error(
                "Could not load demo_samples.json"
            );
        }

        samples =
            await response.json();


        console.log(
            "ONNX model loaded."
        );

        console.log(
            "Input names:",
            session.inputNames
        );

        console.log(
            "Output names:",
            session.outputNames
        );


        statusElement.textContent =
            "Model ready.";


        // Start automatic ECG demo
        await runDemoLoop();

    } catch (error) {

        console.error(error);

        statusElement.textContent =
            "Failed to load model.";
    }
}


// ---------------------------------------
// ONNX inference
// ---------------------------------------

async function classifyECG(signal) {

    const inputData =
        Float32Array.from(signal);


    const inputTensor =
        new ort.Tensor(
            "float32",
            inputData,
            [1, 1, 187]
        );


    const feeds = {
        ecg: inputTensor
    };


    const results =
        await session.run(feeds);


    const logits =
        Array.from(
            results.logits.data
        );


    const probabilities =
        softmax(logits);


    const maxProbability =
        Math.max(...probabilities);


    const predictedClass =
        probabilities.indexOf(
            maxProbability
        );


    return {
        classIndex:
            predictedClass,

        className:
            LABELS[predictedClass],

        confidence:
            maxProbability,

        probabilities:
            probabilities
    };
}


// ---------------------------------------
// Softmax
// ---------------------------------------

function softmax(logits) {

    const maxLogit =
        Math.max(...logits);


    const expValues =
        logits.map(
            value =>
                Math.exp(
                    value - maxLogit
                )
        );


    const expSum =
        expValues.reduce(
            (sum, value) =>
                sum + value,
            0
        );


    return expValues.map(
        value =>
            value / expSum
    );
}


// ---------------------------------------
// Display class probabilities
// ---------------------------------------

function displayProbabilities(
    probabilities
) {

    probabilityBars.innerHTML = "";


    probabilities.forEach(
        (probability, index) => {

            const percentage =
                probability * 100;


            // Entire row
            const row =
                document.createElement(
                    "div"
                );

            row.className =
                "probability-row";


            // Class name
            const label =
                document.createElement(
                    "span"
                );

            label.className =
                "probability-label";

            label.textContent =
                LABELS[index];


            // Bar background
            const barContainer =
                document.createElement(
                    "div"
                );

            barContainer.className =
                "probability-bar-container";


            // Actual probability bar
            const bar =
                document.createElement(
                    "div"
                );

            bar.className =
                "probability-bar";

            bar.style.width =
                `${percentage}%`;


            // Numeric probability
            const value =
                document.createElement(
                    "span"
                );

            value.className =
                "probability-value";

            value.textContent =
                `${percentage.toFixed(1)}%`;


            barContainer.appendChild(
                bar
            );

            row.appendChild(
                label
            );

            row.appendChild(
                barContainer
            );

            row.appendChild(
                value
            );

            probabilityBars.appendChild(
                row
            );
        }
    );
}


// ---------------------------------------
// Random ECG selection
// ---------------------------------------

function getRandomSample() {

    let randomIndex;


    do {

        randomIndex =
            Math.floor(
                Math.random() *
                samples.length
            );

    } while (

        samples.length > 1 &&

        randomIndex ===
        previousSampleIndex

    );


    previousSampleIndex =
        randomIndex;


    return samples[randomIndex];
}


// ---------------------------------------
// Clear prediction display
// ---------------------------------------

function clearResults() {

    predictionElement.textContent =
        "—";

    confidenceElement.textContent =
        "—";

    trueLabelElement.textContent =
        "—";

    probabilityBars.innerHTML =
        "";
}


// ---------------------------------------
// Clear ECG canvas
// ---------------------------------------

function clearCanvas() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
}


// ---------------------------------------
// Animate ECG waveform
// ---------------------------------------

function animateECG(signal) {

    return new Promise(resolve => {

        clearCanvas();


        const width =
            canvas.width;

        const height =
            canvas.height;

        const padding = 30;


        const minValue =
            Math.min(...signal);

        const maxValue =
            Math.max(...signal);


        const valueRange =
            maxValue -
            minValue || 1;


        // Convert signal value
        // into canvas Y position
        function scaleY(value) {

            const normalized =
                (value - minValue) /
                valueRange;


            return (
                height -
                padding -
                normalized *
                (
                    height -
                    2 * padding
                )
            );
        }


        const xStep =
            (
                width -
                2 * padding
            ) /
            (
                signal.length -
                1
            );


        let pointIndex = 0;


        // Approximate total animation:
        // 2.5 seconds
        const animationDuration =
            2500;


        const pointDelay =
            animationDuration /
            signal.length;


        // ECG line appearance
        ctx.beginPath();

        ctx.strokeStyle = "#000000";

        ctx.lineWidth = 2;

        ctx.lineJoin =
            "round";

        ctx.lineCap =
            "round";


        function drawNextPoint() {

            const x =
                padding +
                pointIndex *
                xStep;


            const y =
                scaleY(
                    signal[
                        pointIndex
                    ]
                );


            if (
                pointIndex === 0
            ) {

                ctx.moveTo(
                    x,
                    y
                );

            } else {

                ctx.lineTo(
                    x,
                    y
                );

                ctx.stroke();
            }


            pointIndex++;


            if (
                pointIndex <
                signal.length
            ) {

                setTimeout(
                    drawNextPoint,
                    pointDelay
                );

            } else {

                resolve();
            }
        }


        drawNextPoint();
    });
}


// ---------------------------------------
// Pause helper
// ---------------------------------------

function sleep(milliseconds) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );
}


// ---------------------------------------
// Main automatic demo loop
// ---------------------------------------

async function runDemoLoop() {

    while (true) {

        // -------------------------------
        // Select random heartbeat
        // -------------------------------

        const sample =
            getRandomSample();


        // -------------------------------
        // Clear previous result
        // -------------------------------

        clearResults();


        statusElement.textContent =
            "Displaying heartbeat...";


        // -------------------------------
        // Animate all 187 ECG points
        // -------------------------------

        await animateECG(
            sample.signal
        );


        // -------------------------------
        // Run CNN after animation
        // -------------------------------

        statusElement.textContent =
            "Classifying heartbeat...";


        const result =
            await classifyECG(
                sample.signal
            );


        // -------------------------------
        // Display prediction
        // -------------------------------

        predictionElement.textContent =
            result.className;


        confidenceElement.textContent =
            `${(
                result.confidence *
                100
            ).toFixed(1)}%`;


        trueLabelElement.textContent =
            sample.label_name;


        // Display all five probabilities
        displayProbabilities(
            result.probabilities
        );


        // -------------------------------
        // Classification status
        // -------------------------------

        if (
            result.className ===
            sample.label_name
        ) {

            statusElement.textContent =
                "Correct classification";

        } else {

            statusElement.textContent =
                "Classification complete";
        }


        console.log({
            predicted:
                result.className,

            confidence:
                result.confidence,

            actual:
                sample.label_name,

            probabilities:
                result.probabilities
        });


        // -------------------------------
        // Keep prediction on screen
        // -------------------------------

        await sleep(3500);


        // -------------------------------
        // Transition to next heartbeat
        // -------------------------------

        clearResults();

        clearCanvas();


        statusElement.textContent =
            "Loading next heartbeat...";


        await sleep(700);
    }
}


// ---------------------------------------
// Start application
// ---------------------------------------

initialize();