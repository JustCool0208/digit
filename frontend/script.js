const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
canvas.width = 280;
canvas.height = 280;
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 8;
ctx.lineJoin = "round";
let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!drawing) return;
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.offsetX, event.offsetY);
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
async function sendData() {
    const operation = document.getElementById("operation").value;
    const secondNumber = document.getElementById("secondNumber").value;
    const imageBase64 = canvas.toDataURL();

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageBase64, operation: operation, second_number: parseInt(secondNumber) })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById("result").innerText = `First digit : ${data.first_digit} , Second digit :  ${data.second_digit} , RESULT = ${data.result}`;
    } catch (error) {
        console.error("Error:", error);
    }
}

