document.getElementById("send-btn").addEventListener("click", async () => {
    const message = document.getElementById("message-input").value;
    const selectedFunction = document.getElementById("function-select").value;
    const chatContainer = document.getElementById("chat-container");

    if (!message) return;

    // Show user message
    const userMsg = document.createElement("div");
    userMsg.innerHTML = `<b>User:</b> ${message}`;
    chatContainer.appendChild(userMsg);

    let url = "";

    if (selectedFunction === "answer") {
        url = "/answer";
    } else if (selectedFunction === "kbanswer") {
        url = "/kbanswer";
    } else if (selectedFunction === "search") {
        url = "/search";
    }

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        const botMsg = document.createElement("div");
        botMsg.innerHTML = `<b>Chatbot:</b> ${data.message}`;
        chatContainer.appendChild(botMsg);

    } catch (error) {
        console.error(error);
    }

    document.getElementById("message-input").value = "";
});