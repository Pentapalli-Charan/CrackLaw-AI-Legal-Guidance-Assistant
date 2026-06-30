import { motion } from 'framer-motion';
import {
  Scale, Send, RotateCcw, Share2, BookOpen,
  CheckCircle2, Gavel, FileText, ExternalLink,
} from 'lucide-react';
import './DemoPreview.css';

export default function DemoPreview() {
  return (
    <section className="demo-preview section" id="demo">
      <div className="demo-glow" />

      <div className="container">
        <motion.div
          className="demo-preview-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <span className="section-label">AI Assistant</span>
          <h2 className="section-title">Experience CrackLaw AI</h2>
          <p className="section-subtitle">
            A preview of our intelligent legal assistant — combining conversational AI
            with deep legal knowledge for instant, reliable guidance.
          </p>
        </motion.div>

        <motion.div
          className="demo-chat-container"
          initial={{ opacity: 0, y: 40, scale: 0.97 }}
          whileInView={{ opacity: 1, y: 0, scale: 1 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.7, ease: [0.4, 0, 0.2, 1] }}
        >
          {/* Top Bar */}
          <div className="demo-topbar">
            <div className="demo-topbar-left">
              <div className="demo-topbar-avatar">
                <Scale size={16} color="#ffffff" />
              </div>
              <div>
                <div className="demo-topbar-title">CrackLaw AI Assistant</div>
                <div className="demo-topbar-subtitle">Legal AI • Always Online</div>
              </div>
            </div>
            <div className="demo-topbar-actions">
              <button aria-label="Reset"><RotateCcw size={14} /></button>
              <button aria-label="Share"><Share2 size={14} /></button>
            </div>
          </div>

          {/* Messages */}
          <div className="demo-messages">
            {/* User Message */}
            <div className="demo-msg user">
              <div className="demo-msg-avatar">U</div>
              <div className="demo-msg-content">
                <div className="demo-msg-bubble">
                  What are the legal requirements for drafting a valid employment contract in India?
                </div>
              </div>
            </div>

            {/* AI Response */}
            <div className="demo-msg ai">
              <div className="demo-msg-avatar">
                <Scale size={14} />
              </div>
              <div className="demo-msg-content">
                <div className="demo-msg-bubble">
                  Here&apos;s a comprehensive overview of the key legal requirements for drafting a valid employment contract in India:
                </div>

                {/* Response Card */}
                <div className="demo-response-card">
                  <div className="demo-response-card-title">
                    <BookOpen size={14} />
                    Employment Contract Requirements — India
                  </div>
                  <div className="demo-response-card-items">
                    <div className="demo-response-card-item">
                      <CheckCircle2 size={14} />
                      <span><strong>Essential Clauses:</strong> Job title, compensation, work hours, leave policy, termination terms, and notice period must be clearly stated.</span>
                    </div>
                    <div className="demo-response-card-item">
                      <CheckCircle2 size={14} />
                      <span><strong>Governing Laws:</strong> Indian Contract Act 1872, Industrial Disputes Act 1947, and applicable state-specific Shops & Establishment Acts.</span>
                    </div>
                    <div className="demo-response-card-item">
                      <Gavel size={14} />
                      <span><strong>Key Precedent:</strong> Central Inland Water Transport Corp v. Brojo Nath Ganguly — unconscionable terms are unenforceable.</span>
                    </div>
                    <div className="demo-response-card-item">
                      <FileText size={14} />
                      <span><strong>Non-Compete:</strong> Post-employment non-compete clauses are generally unenforceable under Section 27 of Indian Contract Act.</span>
                    </div>
                  </div>
                </div>

                {/* Source Tags */}
                <div className="demo-sources">
                  <span className="demo-source-tag"><ExternalLink size={10} /> Indian Contract Act</span>
                  <span className="demo-source-tag"><ExternalLink size={10} /> Industrial Disputes Act</span>
                  <span className="demo-source-tag"><ExternalLink size={10} /> Supreme Court DB</span>
                </div>
              </div>
            </div>

            {/* Typing Indicator */}
            <div className="demo-typing">
              <div className="demo-msg-avatar" style={{ background: 'var(--gold-gradient)' }}>
                <Scale size={14} color="#ffffff" />
              </div>
              <div className="demo-typing-dots">
                <span /><span /><span />
              </div>
            </div>
          </div>

          {/* Suggested Prompts */}
          <div className="demo-suggestions">
            <div className="demo-suggestion-pill">Draft a contract template</div>
            <div className="demo-suggestion-pill">Explain Section 302 IPC</div>
            <div className="demo-suggestion-pill">Tenant rights in Maharashtra</div>
            <div className="demo-suggestion-pill">Company registration process</div>
          </div>

          {/* Input Bar */}
          <div className="demo-input-bar">
            <div className="demo-input-field">Ask CrackLaw AI anything about law...</div>
            <button className="demo-input-send" aria-label="Send">
              <Send size={18} color="#ffffff" />
            </button>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
