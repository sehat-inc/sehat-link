import { useEffect, useState } from "react";

function App() {
  const [socket, setSocket] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/chat");
    ws.onmessage = (e) => {
      setMessages(prev => [...prev, { from: "agent", text: e.data }]);
    };
    setSocket(ws);
  }, []);

  const send = () => {
    socket.send(input);
    setMessages(prev => [...prev, { from: "user", text: input }]);
    setInput("");
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Sehat-Link Chat</h2>
      <div style={{ border: "1px solid #ddd", padding: 10, height: 300, overflowY: "auto" }}>
        {messages.map((m, i) => (
          <div key={i}><b>{m.from}:</b> {m.text}</div>
        ))}
      </div>

      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={send}>Send</button>
    </div>
  );
}

export default App;
