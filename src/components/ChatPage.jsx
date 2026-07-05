import { useState, useRef, useEffect, useCallback, memo } from 'react';
import { Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Scale, Send, RotateCcw, Plus, Trash2, MessageSquare,
  Copy, Check, Square, ArrowLeft, AlertCircle,
  ExternalLink, Menu, ShieldCheck, ChevronRight,
} from 'lucide-react';
import useChatStore from '../lib/chatStore';
import './ChatPage.css';

// ── Suggested Prompts ──
const SUGGESTED_PROMPTS = [
  'Explain Section 302 of IPC',
  'Draft a contract template',
  'Tenant rights in Maharashtra',
  'Company registration process',
  'What is bail under CrPC?',
  'Consumer protection remedies',
];

// ── Code Block with copy ──
const CodeBlock = memo(function CodeBlock({ language, children }) {
  const [copied, setCopied] = useState(false);
  const code = String(children).replace(/\n$/, '');

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div>
      <div className="chat-code-header">
        <span>{language || 'code'}</span>
        <button className="chat-code-copy" onClick={handleCopy}>
          {copied ? <><Check size={10} /> Copied</> : <><Copy size={10} /> Copy</>}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneLight}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, padding: '14px', fontSize: '0.8rem', background: '#fafbfc' }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
});

// ── Markdown renderer components ──
const markdownComponents = {
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '');
    const isBlock = props.node?.position?.start?.line !== props.node?.position?.end?.line ||
                    String(children).includes('\n');

    if (match || isBlock) {
      return <CodeBlock language={match?.[1]}>{children}</CodeBlock>;
    }
    return <code className={className} {...props}>{children}</code>;
  },
};

// ── Confidence Badge ──
function ConfidenceBadge({ score }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  let level = 'low';
  if (pct >= 80) level = 'high';
  else if (pct >= 50) level = 'medium';

  return (
    <span className={`chat-confidence-badge ${level}`}>
      <ShieldCheck size={12} />
      {pct}% confidence
    </span>
  );
}

// ── Citations ──
function Citations({ citations }) {
  if (!citations || citations.length === 0) return null;
  return (
    <div className="demo-sources">
      {citations.map((c, i) => (
        <span key={i} className="demo-source-tag">
          <ExternalLink size={10} />
          {c.source || c.act || c.text?.slice(0, 30) || `Source ${i + 1}`}
          {c.section ? ` §${c.section}` : ''}
        </span>
      ))}
    </div>
  );
}

// ── Single Message ──
const ChatMessage = memo(function ChatMessage({ message, onRegenerate, isLast }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (message.role === 'user') {
    return (
      <div className="demo-msg user">
        <div className="demo-msg-avatar">U</div>
        <div className="demo-msg-content">
          <div className="demo-msg-bubble">{message.content}</div>
        </div>
      </div>
    );
  }

  // Assistant message
  const hasError = message.error && !message.content;

  if (hasError) {
    return (
      <div className="demo-msg ai">
        <div className="demo-msg-avatar">
          <Scale size={14} />
        </div>
        <div className="demo-msg-content">
          <div className="chat-error-msg">
            <AlertCircle size={16} />
            <div>
              <div>{message.error}</div>
              {onRegenerate && (
                <button className="chat-error-retry" onClick={onRegenerate}>
                  <RotateCcw size={12} /> Retry
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="demo-msg ai">
      <div className="demo-msg-avatar">
        <Scale size={14} />
      </div>
      <div className="demo-msg-content">
        <div className="demo-msg-bubble">
          <div className="chat-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content || ''}
            </ReactMarkdown>
            {message.isStreaming && <span className="streaming-cursor" />}
          </div>
        </div>

        {!message.isStreaming && message.content && (
          <>
            <Citations citations={message.citations} />
            <ConfidenceBadge score={message.confidence_score} />
            <div className="chat-msg-actions">
              <button
                className={`chat-msg-action-btn ${copied ? 'copied' : ''}`}
                onClick={handleCopy}
              >
                {copied ? <><Check size={11} /> Copied</> : <><Copy size={11} /> Copy</>}
              </button>
              {isLast && onRegenerate && (
                <button className="chat-msg-action-btn" onClick={onRegenerate}>
                  <RotateCcw size={11} /> Regenerate
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
});

// ── Typing Indicator ──
function TypingIndicator() {
  return (
    <div className="demo-typing">
      <div className="demo-msg-avatar" style={{ background: 'var(--gold-gradient)' }}>
        <Scale size={14} color="#ffffff" />
      </div>
      <div className="demo-typing-dots">
        <span /><span /><span />
      </div>
    </div>
  );
}

// ── Time formatter ──
function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  const now = new Date();
  const diff = now - d;

  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return d.toLocaleDateString();
}

// ═══════════════════════════════════════════
// ── Main ChatPage Component ──
// ═══════════════════════════════════════════

export default function ChatPage() {
  const {
    sessions, activeSessionId, isStreaming, conversations,
    createSession, switchSession, deleteSession,
    sendMessage, stopGeneration, regenerate,
  } = useChatStore();

  const [input, setInput] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const messagesAreaRef = useRef(null);

  const messages = conversations[activeSessionId] || [];

  // ── Auto-scroll ──
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // ── Auto-focus input ──
  useEffect(() => {
    if (!isStreaming) {
      inputRef.current?.focus();
    }
  }, [isStreaming, activeSessionId]);

  // ── Send handler ──
  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    setInput('');
    sendMessage(trimmed);
  }, [input, isStreaming, sendMessage]);

  // ── Keyboard handler ──
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // ── Suggestion click ──
  const handleSuggestion = useCallback((prompt) => {
    if (isStreaming) return;
    setInput('');
    sendMessage(prompt);
  }, [isStreaming, sendMessage]);

  // ── Auto-resize textarea ──
  const handleInputChange = useCallback((e) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
  }, []);

  // ── Delete session with confirmation ──
  const handleDelete = useCallback((e, sessionId) => {
    e.stopPropagation();
    deleteSession(sessionId);
  }, [deleteSession]);

  const showWelcome = messages.length === 0;
  const showTyping = isStreaming && messages.length > 0 &&
    messages[messages.length - 1]?.role === 'assistant' &&
    !messages[messages.length - 1]?.content;

  return (
    <div className="chat-page">
      {/* Mobile overlay */}
      <div
        className={`chat-sidebar-overlay ${sidebarOpen ? 'open' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Sidebar */}
      <aside className={`chat-sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="chat-sidebar-header">
          <Link to="/" className="chat-sidebar-brand">
            <div className="chat-sidebar-brand-icon">
              <Scale size={16} color="#ffffff" />
            </div>
            <span>CrackLaw</span>
          </Link>
          <button
            className="chat-sidebar-new"
            onClick={() => { createSession(); setSidebarOpen(false); }}
            id="new-chat-btn"
          >
            <Plus size={16} /> New Chat
          </button>
        </div>

        <div className="chat-sidebar-sessions">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`chat-sidebar-session ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => { switchSession(session.id); setSidebarOpen(false); }}
            >
              <MessageSquare size={16} style={{ flexShrink: 0, color: 'var(--text-muted)' }} />
              <div className="chat-sidebar-session-info">
                <div className="chat-sidebar-session-title">{session.title}</div>
                <div className="chat-sidebar-session-time">{formatTime(session.updatedAt)}</div>
              </div>
              <button
                className="chat-sidebar-session-delete"
                onClick={(e) => handleDelete(e, session.id)}
                aria-label="Delete session"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <div style={{ padding: '20px 12px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              No conversations yet
            </div>
          )}
        </div>

        <Link to="/" className="chat-sidebar-back">
          <ArrowLeft size={14} /> Back to Home
        </Link>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Top Bar — reuses demo-topbar pattern */}
        <div className="demo-topbar">
          <div className="demo-topbar-left">
            <button
              className="chat-mobile-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle sidebar"
            >
              <Menu size={18} />
            </button>
            <div className="demo-topbar-avatar">
              <Scale size={16} color="#ffffff" />
            </div>
            <div>
              <div className="demo-topbar-title">CrackLaw AI Assistant</div>
              <div className="demo-topbar-subtitle">Legal AI • Always Online</div>
            </div>
          </div>
          <div className="demo-topbar-actions">
            <button
              aria-label="New chat"
              title="New chat"
              onClick={() => createSession()}
              style={{ cursor: 'pointer' }}
            >
              <Plus size={14} />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="chat-messages-area" ref={messagesAreaRef}>
          {showWelcome ? (
            <div className="chat-welcome">
              <div className="chat-welcome-icon">
                <Scale size={28} color="#ffffff" />
              </div>
              <h2>How can I help you today?</h2>
              <p>
                Ask me anything about Indian law — legal research, contract analysis,
                case precedents, statutory interpretation, and more.
              </p>
              <div className="chat-welcome-prompts">
                {SUGGESTED_PROMPTS.map((prompt) => (
                  <div
                    key={prompt}
                    className="demo-suggestion-pill"
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleSuggestion(prompt)}
                  >
                    {prompt} <ChevronRight size={12} />
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <ChatMessage
                  key={`${activeSessionId}-${i}`}
                  message={msg}
                  isLast={i === messages.length - 1 && msg.role === 'assistant'}
                  onRegenerate={
                    i === messages.length - 1 && msg.role === 'assistant' && !msg.isStreaming
                      ? regenerate
                      : null
                  }
                />
              ))}
              {showTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="chat-input-wrapper">
          {/* Suggestion pills — show only when there are messages */}
          {!showWelcome && !isStreaming && messages.length > 0 && messages.length < 4 && (
            <div className="chat-input-suggestions">
              {SUGGESTED_PROMPTS.slice(0, 3).map((prompt) => (
                <div
                  key={prompt}
                  className="demo-suggestion-pill"
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleSuggestion(prompt)}
                >
                  {prompt}
                </div>
              ))}
            </div>
          )}

          <div className="chat-input-bar">
            <textarea
              ref={inputRef}
              className="chat-input-field"
              placeholder="Ask CrackLaw AI anything about law..."
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              rows={1}
              id="chat-input"
            />
            {isStreaming ? (
              <button
                className="chat-input-send stop-btn"
                onClick={stopGeneration}
                aria-label="Stop generation"
                id="stop-btn"
              >
                <Square size={18} color="#ffffff" />
              </button>
            ) : (
              <button
                className="chat-input-send"
                onClick={handleSend}
                disabled={!input.trim()}
                aria-label="Send message"
                id="send-btn"
              >
                <Send size={18} color="#ffffff" />
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
