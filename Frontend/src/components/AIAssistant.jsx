import React, { useEffect, useMemo, useRef, useState } from "react";
import { FiX } from "react-icons/fi";
import { PiRobotLight } from "react-icons/pi";
import "./AIAssistant.css";

export default function AIAssistant({ onSend, context, initialMessages = [], onClose }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState(
    (initialMessages || []).map((m) =>
      typeof m === "string"
        ? { role: "ai", content: m }
        : { role: m.role || "ai", content: m.content }
    )
  );
  const [sending, setSending] = useState(false);
  const [showHint, setShowHint] = useState(() => !localStorage.getItem("ai_hint_seen"));
  const bodyRef = useRef(null);

  const speak = (text) => {
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1; u.pitch = 1; u.lang = "en-US";
      window.speechSynthesis.speak(u);
    }
  };

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: "smooth" });
    });
  };
  useEffect(scrollToBottom, [messages, open]);

  useEffect(() => {
    if (!showHint) return;
    const t = setTimeout(() => {
      setShowHint(false);
      localStorage.setItem("ai_hint_seen", "1");
    }, 5500);
    return () => clearTimeout(t);
  }, [showHint]);

  const quickPrompts = useMemo(
    () => [
      "Why was this alert raised?",
      "What’s the likely root cause?",
      "What should I do first?",
      "Which sensor is most suspicious?",
      "How do I verify the fix worked?",
    ],
    []
  );

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");

    try {
      const reply = await onSend(text, context);
      const content = typeof reply === "string" ? reply : reply?.reply || "🤖 No reply.";
      setMessages((m) => [...m, { role: "ai", content }]);
      speak(content);
    } catch {
      setMessages((m) => [...m, { role: "ai", content: "⚠️ Error calling AI." }]);
    } finally {
      setSending(false);
    }
  };

  const dismissHint = () => {
    setShowHint(false);
    localStorage.setItem("ai_hint_seen", "1");
  };

  return (
    <div className="ai-assistant-container">
      {!open && (
        <div className="ai-fab-wrap">
          {showHint && (
            <div className="ai-hint" role="status" aria-live="polite">
              <span className="ai-hint-text">Ask me questions</span>
            
            </div>
          )}

          <button
            className="ai-fab"
            onClick={() => { setOpen(true); dismissHint(); }}
            title="Talk to AI"
            onMouseEnter={() => setShowHint(true)}
            onMouseLeave={() => setShowHint(false)}
            onFocus={() => setShowHint(true)}
            onBlur={() => setShowHint(false)}
          >
            <span className="ai-avatar" aria-hidden="true">
              <span className="ai-orb" />
              <PiRobotLight size={18} className="ai-robot" />
              <span className="ai-ping" />
            </span>
            <span className="ai-fab-label">AI Assistant</span>
          </button>
        </div>
      )}

      {open && (
        <div className="ai-chatbox" role="dialog" aria-label="AI Assistant">
          <div className="chat-header">
            <div className="title">
              <span className="ai-avatar lg" aria-hidden="true">
                <span className="ai-orb" />
                <PiRobotLight size={18} className="ai-robot" />
              </span>
              <span>AI Assistant</span>
            </div>
            <button
              className="icon-btn"
              onClick={() => { setOpen(false); onClose?.(); }}
              aria-label="Close assistant"
            >
              <FiX size={18} />
            </button>
          </div>

          <div className="chat-body" ref={bodyRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role}`}>
                {msg.content}
              </div>
            ))}
          </div>

          <div className="quick-prompts" aria-label="Quick questions">
            {quickPrompts.map((q) => (
              <button
                key={q}
                className="qpill"
                onClick={() => setInput(q)}
                disabled={sending}
              >
                {q}
              </button>
            ))}
          </div>

          <div className="chat-input">
            <input
              type="text"
              value={input}
              placeholder="Ask anything about this machine..."
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={sending}
              aria-label="Your message"
            />
            <button onClick={handleSend} disabled={sending} className="send-btn">
              {sending ? "…" : "Send"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
