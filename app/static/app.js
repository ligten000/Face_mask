
const API_BASE = window.location.origin;

const video = document.getElementById("webcam");
const canvas = document.getElementById("capture-canvas");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");
const placeholder = document.getElementById("camera-placeholder");

let stream = null;
let running = false;
let sending = false;
let timerId = null;

let latestPredictions = [];
let total = 0;
let correct = 0;
let without = 0;
let incorrect = 0;

const history = [];
const MAX_HISTORY = 20;

function setApiStatus(online) {
    document.getElementById("api-status").textContent =
        online ? "API đang hoạt động" : "API mất kết nối";

    document.getElementById("backend-status").textContent =
        online ? "Đang hoạt động" : "Không kết nối";

    document.getElementById("sidebar-status").textContent =
        online ? "Server đã kết nối" : "Server ngoại tuyến";

    document.getElementById("api-dot").className =
        "status-dot " + (online ? "online" : "offline");
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (!response.ok) throw new Error("Health check failed");

        const data = await response.json();
        setApiStatus(data.status === "ok");
    } catch {
        setApiStatus(false);
    }
}

function renderStats() {
    document.getElementById("total-count").textContent = total;
    document.getElementById("correct-count").textContent = correct;
    document.getElementById("without-count").textContent = without;
    document.getElementById("incorrect-count").textContent = incorrect;
}

function renderHistory() {
    const tbody = document.getElementById("history-body");
    tbody.replaceChildren();

    if (history.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 3;
        cell.className = "empty-state";
        cell.textContent = "Chưa có kết quả AI để hiển thị.";

        row.appendChild(cell);
        tbody.appendChild(row);
        return;
    }

    history.slice(0, MAX_HISTORY).forEach(item => {
        const row = document.createElement("tr");

        [item.time, item.label, item.confidence].forEach(value => {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });
}

function getLabelInfo(prediction) {
    const raw = String(prediction.label ?? "").toLowerCase();

    if (raw === "with_mask") {
        return {
            key: "correct",
            text: "Đeo khẩu trang đúng cách"
        };
    }

    if (raw === "without_mask") {
        return {
            key: "without",
            text: "Không đeo khẩu trang"
        };
    }

    if (raw === "with_incorrect_mask") {
        return {
            key: "incorrect",
            text: "Đeo khẩu trang không đúng cách"
        };
    }

    return null;
}

function formatConfidence(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return "Không có dữ liệu";
    }

    const percentage = value <= 1 ? value * 100 : value;
    return `${percentage.toFixed(1)}%`;
}

function getBoxColor(key) {
    if (key === "correct") return "#22c55e";
    if (key === "without") return "#ef4444";
    return "#f59e0b";
}


function drawPredictions(predictions) {
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    if (!videoWidth || !videoHeight) return;

    // Canvas sử dụng cùng hệ tọa độ với video gốc.
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const colorMap = {
        correct: "#22c55e",
        without: "#ef4444",
        incorrect: "#f59e0b"
    };

    ctx.lineWidth = Math.max(2, videoWidth / 320);
    ctx.font = `bold ${Math.max(14, videoWidth / 35)}px Arial`;

    predictions.forEach(prediction => {
        const box = prediction.bbox;
        const info = getLabelInfo(prediction);

        if (!box || !info) return;

        // API hiện nhận ảnh 640 × 480.
        const scaleX = videoWidth / 640;
        const scaleY = videoHeight / 480;

        const x = Number(box.x) * scaleX;
        const y = Number(box.y) * scaleY;
        const w = Number(box.width) * scaleX;
        const h = Number(box.height) * scaleY;

        if (![x, y, w, h].every(Number.isFinite)) return;

        const color = colorMap[info.key];
        const label =
            `${info.text} (${formatConfidence(prediction.confidence)})`;

        ctx.strokeStyle = color;
        ctx.strokeRect(x, y, w, h);

        const labelHeight = Math.max(24, videoWidth / 22);
        const labelWidth = Math.min(
            ctx.measureText(label).width + 12,
            videoWidth - x
        );
        const labelY = Math.max(labelHeight, y);

        ctx.fillStyle = color;
        ctx.fillRect(
            x,
            labelY - labelHeight,
            labelWidth,
            labelHeight
        );

        ctx.fillStyle = "#ffffff";
        ctx.fillText(
            label,
            x + 6,
            labelY - labelHeight / 2 + 5
        );
    });
}


function processPredictions(predictions) {
    latestPredictions = Array.isArray(predictions) ? predictions : [];
    drawPredictions(latestPredictions);

    if (latestPredictions.length === 0) {
        document.getElementById("detection-message").textContent =
            "Chưa phát hiện khuôn mặt trong khung hình.";
        return;
    }

    // Mỗi phản hồi là trạng thái mới nhất của camera.
    // Các bộ đếm biểu thị số khuôn mặt theo kết quả hiện tại,
    // không phải tổng số lượt qua tất cả các khung hình.
    correct = 0;
    without = 0;
    incorrect = 0;

    latestPredictions.forEach(prediction => {
        const info = getLabelInfo(prediction);
        if (!info) return;

        if (info.key === "correct") correct++;
        else if (info.key === "without") without++;
        else incorrect++;
    });

    total = correct + without + incorrect;
    renderStats();

    // Chỉ thêm vào lịch sử khi nhãn hoặc bbox thay đổi,
    // tránh ghi lại liên tục cùng một khuôn mặt mỗi giây.
    const currentSignature = latestPredictions.map(prediction => {
        const box = prediction.bbox || {};
        return [
            prediction.label,
            box.x,
            box.y,
            box.width,
            box.height
        ].join(":");
    }).join("|");

    if (currentSignature !== processPredictions.lastSignature) {
        latestPredictions.forEach(prediction => {
            const info = getLabelInfo(prediction);
            if (!info) return;

            history.unshift({
                time: new Date().toLocaleTimeString("vi-VN"),
                label: info.text,
                confidence: formatConfidence(prediction.confidence)
            });
        });

        history.splice(MAX_HISTORY);
        renderHistory();
        processPredictions.lastSignature = currentSignature;
    }

    document.getElementById("detection-message").textContent =
        `Đang nhận diện ${total} khuôn mặt.`;
}

processPredictions.lastSignature = "";

async function sendFrame() {
    if (!running || sending || video.readyState < 2) return;

    sending = true;

    try {
        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = 640;
        captureCanvas.height = 480;

        const captureContext = captureCanvas.getContext("2d");
        captureContext.drawImage(
            video,
            0,
            0,
            captureCanvas.width,
            captureCanvas.height
        );

        const blob = await new Promise(resolve => {
            captureCanvas.toBlob(resolve, "image/jpeg", 0.75);
        });

        if (!blob) throw new Error("Không tạo được ảnh webcam.");

        const formData = new FormData();
        formData.append("file", blob, "webcam.jpg");

        const response = await fetch(`${API_BASE}/api/detect`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API trả về HTTP ${response.status}`);
        }

        const result = await response.json();
        processPredictions(result.predictions);

    } catch (error) {
        document.getElementById("detection-message").textContent =
            `Lỗi gửi ảnh: ${error.message}`;
    } finally {
        sending = false;
    }
}

async function captureLoop() {
    if (!running) return;

    await sendFrame();

    if (running) {
        timerId = setTimeout(captureLoop, 1000);
    }
}

startButton.addEventListener("click", async () => {
    if (running) return;

    if (!navigator.mediaDevices?.getUserMedia) {
        alert("Trình duyệt không hỗ trợ webcam hoặc trang chưa chạy trên HTTPS/localhost.");
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        running = true;
        video.style.display = "block";
        placeholder.style.display = "none";

        canvas.hidden = false;
        canvas.style.display = "block";

        startButton.disabled = true;
        stopButton.disabled = false;

        document.getElementById("camera-status").textContent =
            "Đang hoạt động";

        captureLoop();

    } catch (error) {
        alert(
            "Không mở được webcam. Hãy kiểm tra camera và quyền truy cập.\n" +
            error.message
        );
    }
});

stopButton.addEventListener("click", () => {
    running = false;

    if (timerId !== null) {
        clearTimeout(timerId);
        timerId = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    video.style.display = "none";
    canvas.hidden = true;
    canvas.style.display = "none";
    placeholder.style.display = "flex";

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    startButton.disabled = false;
    stopButton.disabled = true;

    document.getElementById("camera-status").textContent = "Đã tắt";
    document.getElementById("detection-message").textContent =
        "Đã dừng webcam.";
});

document.getElementById("clear-history").addEventListener("click", () => {
    history.length = 0;
    total = 0;
    correct = 0;
    without = 0;
    incorrect = 0;

    processPredictions.lastSignature = "";

    renderStats();
    renderHistory();

    document.getElementById("detection-message").textContent =
        "Đã xóa thống kê của phiên trình duyệt hiện tại.";
});

checkHealth();
setInterval(checkHealth, 5000);
renderStats();
renderHistory();
