import { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [mode, setMode] = useState("signup"); // 'login' or 'signup'
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState(""); // for signup
  const [token, setToken] = useState("");
  const [loggedIn, setLoggedIn] = useState(false);

  const [session, setSession] = useState(null);
  const [socket, setSocket] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  // Signup handler
  const handleSignup = async () => {
    try {
      const res = await axios.post("http://localhost:8000/patient/signup", {
        email,
        password,
        name, // include any extra fields your backend expects
      });
      alert(`Signup successful! Patient ID: ${res.data.patient_id}`);
      setMode("login"); // switch to login after signup
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.detail || "Signup failed");
    }
  };

  // Login handler
  const handleLogin = async () => {
    try {
      const res = await axios.post("http://localhost:8000/patient/login", {
        email,
        password,
      });

      setToken(res.data.access_token);
      setLoggedIn(true);
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.detail || "Login failed");
    }
  };

  const [userId, setUserId] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  // Start chat session after login
  useEffect(() => {
    if (!token) return;

    const startChat = async () => {
      try {
        const res = await axios.get("http://localhost:8000/patient/chat-start", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setSessionId(res.data.session_id);
        setUserId(res.data.user_id);
        
        console.log("FROM FRONTEND\nSession:", res.data.session_id, "User ID:", res.data.user_id);

        // Open websocket connection
        const ws = new WebSocket("ws://localhost:8000/ws/chat");

        ws.onmessage = (e) => {
          const data = JSON.parse(e.data)
          setMessages((prev) => [...prev, { from: "agent", text: data.response}]);
        };
        
        ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        };
      
        ws.onclose = () => {
          console.log("WebSocket disconnected");
        };
        
        setSocket(ws);
      
      } catch (err) {
        console.error(err);
        alert("Failed to start chat session");
      }
    };

    startChat();
  }, [token]);

  // Send chat message
  const sendMessage = () => {
    if (!input || !socket || socket.readyState !== WebSocket.OPEN) return;

    const payload = { message: input, user_id: userId };
    
    socket.send(JSON.stringify(payload));

    setMessages((prev) => [...prev, { from: "user", text: input }]);
    setInput("");
  };

  // Render signup/login form if not logged in
  if (!loggedIn) {
    return (
      <div style={{ padding: 20 }}>
        <h2>{mode === "login" ? "Patient Login" : "Patient Signup"} (Test)</h2>

        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{ display: "block", marginBottom: 10 }}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ display: "block", marginBottom: 10 }}
        />

        {mode === "signup" && (
          <input
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{ display: "block", marginBottom: 10 }}
          />
        )}

        {mode === "login" ? (
          <button onClick={handleLogin}>Login</button>
        ) : (
          <button onClick={handleSignup}>Signup</button>
        )}

        <div style={{ marginTop: 10 }}>
          <button onClick={() => setMode(mode === "login" ? "signup" : "login")}>
            Switch to {mode === "login" ? "Signup" : "Login"}
          </button>
        </div>
      </div>
    );
  }

  // Chat UI after login
  return (
    <div style={{ padding: 20 }}>
      <h2>Sehat-Link Chat</h2>
      <div
        style={{
          border: "1px solid #ddd",
          padding: 10,
          height: 300,
          overflowY: "auto",
          marginBottom: 10,
        }}
      >
        {messages.map((m, i) => (
          <div key={i}>
            <b>{m.from}:</b> {m.text}
          </div>
        ))}
      </div>

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        style={{ marginRight: 10 }}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default App;
