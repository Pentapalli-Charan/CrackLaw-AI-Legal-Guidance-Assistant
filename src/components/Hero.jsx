import { motion } from 'framer-motion';
import {
  ArrowRight, Sparkles, Scale, Shield, Users, Send,
  BookOpen, CheckCircle2, Gavel
} from 'lucide-react';
import './Hero.css';

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i = 0) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.6, delay: i * 0.15, ease: [0.4, 0, 0.2, 1] }
  }),
};

export default function Hero() {
  return (
    <section className="hero section" id="home">
      {/* Background Effects */}
      <div className="hero-bg-glow hero-bg-glow-1" />
      <div className="hero-bg-glow hero-bg-glow-2" />
      <div className="hero-grid-overlay" />

      {/* Floating decorative elements */}
      <div className="hero-floating hero-float-1">⚖</div>
      <div className="hero-floating hero-float-2">§</div>
      <div className="hero-floating hero-float-3">⚖</div>

      <div className="container hero-inner">
        {/* Left Column */}
        <div className="hero-content">
          <motion.div
            className="hero-badge"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={0}
          >
            <span className="hero-badge-dot" />
            AI-Powered Legal Intelligence
          </motion.div>

          <motion.h1
            className="hero-headline"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={1}
          >
            Your AI Legal<br />
            Advisor, <span className="highlight">Available 24/7</span>
          </motion.h1>

          <motion.p
            className="hero-description"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={2}
          >
            Experience the future of legal consultation. CrackLaw combines advanced AI
            with comprehensive legal databases to deliver instant, accurate legal guidance
            for professionals and individuals alike.
          </motion.p>

          <motion.div
            className="hero-buttons"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={3}
          >
            <button className="btn-primary">
              Start AI Consultation <ArrowRight size={16} />
            </button>
            <button className="btn-secondary">
              <Sparkles size={16} /> Explore Features
            </button>
          </motion.div>

          <motion.div
            className="hero-trust"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={4}
          >
            <div className="hero-trust-item">
              <Shield size={14} /> Bank-Grade Security
            </div>
            <div className="hero-trust-item">
              <Users size={14} /> 50K+ Users
            </div>
            <div className="hero-trust-item">
              <Scale size={14} /> 95% Accuracy
            </div>
          </motion.div>
        </div>

        {/* Right Column — Chat Mockup */}
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, x: 60, scale: 0.95 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
        >
          <div className="hero-chat-mockup">
            <div className="chat-header">
              <div className="chat-header-left">
                <div className="chat-header-avatar">
                  <Scale size={16} color="#ffffff" />
                </div>
                <div className="chat-header-info">
                  <h4>CrackLaw AI</h4>
                  <span>Online</span>
                </div>
              </div>
              <div className="chat-header-dots">
                <span /><span /><span />
              </div>
            </div>

            <div className="chat-body">
              <div className="chat-bubble user">
                Can you explain Section 302 of IPC and its implications in a self-defense case?
              </div>

              <div className="chat-bubble ai">
                I&apos;d be happy to help. Let me analyze Section 302 IPC in the context of self-defense.
              </div>

              <div className="chat-ai-card">
                <div className="chat-ai-card-header">
                  <BookOpen size={14} />
                  Legal Analysis — Section 302 IPC
                </div>
                <div className="chat-ai-card-body">
                  <div className="chat-ai-card-item">
                    <CheckCircle2 size={14} />
                    <span>Section 302 deals with punishment for murder — life imprisonment or death penalty.</span>
                  </div>
                  <div className="chat-ai-card-item">
                    <CheckCircle2 size={14} />
                    <span>Right of private defense under Section 96-106 may apply as exception.</span>
                  </div>
                  <div className="chat-ai-card-item">
                    <Gavel size={14} />
                    <span>Key precedent: <strong>Darshan Singh v. State of Punjab (2010)</strong></span>
                  </div>
                </div>
              </div>
            </div>

            <div className="chat-input-mockup">
              <span>Ask CrackLaw AI anything...</span>
              <button aria-label="Send">
                <Send size={16} color="#ffffff" />
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
