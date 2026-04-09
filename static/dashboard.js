const CLASSES = ["Kem", "Phao", "Bong", "Dam", "Gau", "Tien", "Anh", "Niem"];

// Tạo biểu đồ (không cần giữ lại chart cũ mỗi lần)
let accuracyChart = null;
let profitChart = null;
window.addEventListener('load', async () => {
    createCharts();
    await loadData();
});

async function loadData() {
    try {
        const res = await fetch("/status/json?_=" + Date.now());
        const data = await res.json();
        updateUI(data);
    } catch (err) {
        console.error("Failed to load data", err);
    }
}

function createCharts() {
    const ctxAcc = document.getElementById("accuracyChart").getContext("2d");
    accuracyChart = new Chart(ctxAcc, {
        type: "bar",
        data: {
            labels: CLASSES,
            datasets: [{
                label: "Accuracy %",
                data: CLASSES.map(() => 0),
                backgroundColor: CLASSES.map(() => "rgba(75, 192, 192, 0.5)")
            }]
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } }
        }
    });

    const ctxProfit = document.getElementById("profitChart").getContext("2d");
    profitChart = new Chart(ctxProfit, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Profit",
                data: [],
                borderColor: "#00c3ff",
                fill: true
            }]
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: false } }
        }
    });
}


// Sử dụng SSE để nhận thông báo từ server
const eventSource = new EventSource("/events");

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    console.log("Có dữ liệu mới → cập nhật ngay");

    // Update UI luôn ở đây → KHÔNG gọi loadData nữa
    updateUI(data);
};

function updateUI(data) {
    document.getElementById("total_called").innerText = data.total_called ?? 0;
    document.getElementById("total_correct").innerText = data.total_correct ?? 0;
    document.getElementById("total_profit").innerText = data.total_profit ?? 0;
    document.getElementById("winrate").innerText = (data.winrate ?? 0).toFixed(2) + "%";
    document.getElementById("drift_status").innerText = data.drift_alert ? "🚨 DRIFT ALERT!" : "✅ Normal";
    document.getElementById("drift_status").className = data.drift_alert ? "alert" : "highlight";
    document.getElementById("last_prediction").innerText = (data.last_prediction || []).join(", ");

    const historyBar = document.getElementById("history-bar");
    historyBar.innerHTML = "";

    if (Array.isArray(data.history_results)) {
        data.history_results.forEach((result, i) => {
            const img = document.createElement("img");
            img.src = `/static/img/${String(result).toUpperCase()}.png`;
            img.style.width = "60px";
            img.style.height = "60px";
            img.style.objectFit = "contain";
            img.style.backgroundColor = "#fff";
            img.style.borderRadius = "8px";
            img.style.padding = "5px";
            img.title = result;

            const isCorrect = Array.isArray(data.history_outcomes)
                ? data.history_outcomes[i]
                : null;

            if (isCorrect === true) {
                img.style.border = "3px solid #22c55e";
                img.style.boxShadow = "0 0 10px rgba(34, 197, 94, 0.45)";
            } else if (isCorrect === false) {
                img.style.border = "3px solid #ef4444";
                img.style.boxShadow = "0 0 10px rgba(239, 68, 68, 0.35)";
            } else {
                img.style.border = "1px solid #ccc";
                img.style.boxShadow = "none";
            }

            historyBar.appendChild(img);
        });
    }

    const logBox = document.getElementById("logs");
    const logsToShow = (data.logs || []).slice(0, 50);
    logBox.innerHTML = logsToShow.join("<br>");

    const accuracyData = CLASSES.map(cls => {
        const correct = data.class_correct?.[cls] || 0;
        const called = data.class_called?.[cls] || 0;
        return called > 0 ? +(correct / called * 100).toFixed(2) : 0;
    });

    const accuracyColors = accuracyData.map(v =>
        v >= 50 ? "rgba(75, 192, 192, 0.5)"
        : v >= 20 ? "rgba(255, 159, 64, 0.5)"
        : "rgba(255, 99, 132, 0.5)"
    );

    if (accuracyChart) {
        accuracyChart.data.datasets[0].data = accuracyData;
        accuracyChart.data.datasets[0].backgroundColor = accuracyColors;
        accuracyChart.update("none");
    }

    const profitRaw = data.profit_history || [];
    const cumulativeProfit = [];
    let total = 0;
    for (const p of profitRaw) {
        total += p;
        cumulativeProfit.push(total);
    }

    if (profitChart) {
        profitChart.data.labels = cumulativeProfit.map((_, i) => i + 1);
        profitChart.data.datasets[0].data = cumulativeProfit;
        profitChart.update("none");
    }
}

eventSource.onerror = function() {
    console.error("Mất kết nối SSE, sẽ tự động thử lại...");
};
